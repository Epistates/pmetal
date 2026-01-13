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
    /// Apple7: M1 series
    Apple7,
    /// Apple8: M2 series
    Apple8,
    /// Apple9: M3/M4 series (Dynamic Caching, enhanced ray tracing)
    Apple9,
    /// Apple10: M5 series (Neural Accelerators in GPU cores)
    Apple10,
    /// Unknown family
    Unknown,
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
}

impl DeviceProperties {
    /// Check if this device supports Apple Family 9 features (M3/M4).
    #[inline]
    pub fn is_apple9_or_newer(&self) -> bool {
        self.gpu_family >= AppleGPUFamily::Apple9
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
        match self.device_tier {
            DeviceTier::Ultra | DeviceTier::Max => (64, 64, 32),
            DeviceTier::Pro => (64, 32, 32),
            DeviceTier::Base => (32, 32, 32),
        }
    }
}

/// Detect GPU family from device name.
fn detect_gpu_family(name: &str) -> AppleGPUFamily {
    // M5 series -> Apple10
    if name.contains("M5") {
        return AppleGPUFamily::Apple10;
    }
    // M4 or M3 series -> Apple9
    if name.contains("M4") || name.contains("M3") || name.contains("A17") {
        return AppleGPUFamily::Apple9;
    }
    // M2 series -> Apple8
    if name.contains("M2") || name.contains("A16") || name.contains("A15") {
        return AppleGPUFamily::Apple8;
    }
    // M1 series -> Apple7
    if name.contains("M1") || name.contains("A14") {
        return AppleGPUFamily::Apple7;
    }
    AppleGPUFamily::Unknown
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

        let properties = DeviceProperties {
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
        };

        debug!("Device properties: {:?}", properties);
        info!(
            "GPU Family: {:?}, Tier: {:?}, Dynamic Caching: {}",
            gpu_family, device_tier, is_apple9_plus
        );

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

    /// Check if this device supports the Neural Engine (M5+).
    ///
    /// Neural Engine support requires macOS 26.2+ and an M5 chip.
    #[inline]
    pub fn supports_neural_engine(&self) -> bool {
        // Check for macOS 26.2+ and M5 chip
        // This is a placeholder - actual detection would use system APIs
        self.properties.name.contains("M5")
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
}
