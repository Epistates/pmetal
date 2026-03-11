//! Metal device and command queue management.
//!
//! This module provides thread-safe access to the Metal device and command queue,
//! using a global singleton pattern for efficiency.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
use parking_lot::RwLock;
use std::sync::{Arc, OnceLock};
use tracing::{debug, info};

use crate::error::{MetalError, Result};
use crate::pipeline::PipelineCache;
use crate::tuna::Tuner;

/// Global Metal context singleton using OnceLock for thread-safe lazy initialization.
static GLOBAL_CONTEXT: OnceLock<Result<Arc<MetalContext>>> = OnceLock::new();

/// Embedded Metal library binary (compiled at build time).
const METAL_LIBRARY_BYTES: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/pmetal_kernels.metallib"));

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

    /// Estimated memory bandwidth in GB/s (queried via sysctl or tier-based fallback).
    pub memory_bandwidth_gbps: f64,

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

    /// Get recommended tile size for matrix operations.
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

/// Detect the number of hardware packages (dies) via `sysctl hw.packages`.
/// Returns 2 for UltraFusion chips, 1 for single-die chips.
fn detect_die_count() -> u32 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
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

        // Apple9+ features
        let is_apple9_plus = gpu_family >= AppleGPUFamily::Apple9;

        let mut properties = DeviceProperties {
            name: name.clone(),
            max_threads_per_threadgroup: device.maxThreadsPerThreadgroup().width as u64,
            max_threadgroup_memory_length: device.maxThreadgroupMemoryLength() as u64,
            has_unified_memory: device.hasUnifiedMemory(),
            recommended_working_set_size: device.recommendedMaxWorkingSetSize(),
            max_buffer_length: device.maxBufferLength() as u64,
            gpu_family,
            device_tier,
            has_dynamic_caching: is_apple9_plus,
            has_hardware_ray_tracing: is_apple9_plus,
            has_mesh_shaders: is_apple9_plus,
            has_nax: gpu_family >= AppleGPUFamily::Apple10,
            architecture_gen: architecture_gen(gpu_family),
            memory_bandwidth_gbps: estimate_memory_bandwidth(device_tier, gpu_family),
            gpu_core_count: estimate_gpu_cores(&name, device_tier),
            ane_core_count: estimate_ane_cores(device_tier),
            die_count: 0,           // set below
            is_ultra_fusion: false, // set below
        };
        let die_count = detect_die_count();
        properties.die_count = die_count;
        properties.is_ultra_fusion = die_count > 1;

        debug!("Device properties: {:?}", properties);
        if properties.is_ultra_fusion {
            info!(
                "GPU: {:?} {:?} (gen {}), {} cores ({} dies, UltraFusion), {:.0} GB/s, NAX: {}, ANE: {} cores",
                properties.gpu_family,
                properties.device_tier,
                properties.architecture_gen,
                properties.gpu_core_count,
                properties.die_count,
                properties.memory_bandwidth_gbps,
                properties.has_nax,
                properties.ane_core_count,
            );
        } else {
            info!(
                "GPU: {:?} {:?} (gen {}), {} cores, {:.0} GB/s, NAX: {}, ANE: {} cores",
                properties.gpu_family,
                properties.device_tier,
                properties.architecture_gen,
                properties.gpu_core_count,
                properties.memory_bandwidth_gbps,
                properties.has_nax,
                properties.ane_core_count,
            );
        }

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

        let pipeline_cache = RwLock::new(cache);
        let tuner = Arc::new(Tuner::new());

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
}
