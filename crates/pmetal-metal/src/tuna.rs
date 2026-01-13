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
//! # Example
//!
//! ```ignore
//! let tuner = Tuner::new();
//! let config = tuner.tune_lora_forward(&ctx, batch_size, in_features, out_features, rank)?;
//! println!("Best config: {:?}", config);
//! ```

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::time::Instant;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::context::MetalContext;
use crate::error::{MetalError, Result};
use tracing::{debug, info};

/// Configuration for a tuned kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// The Auto-Tuner.
pub struct Tuner {
    /// Cache of best configurations.
    /// Key: "kernel_name:M:N:K" (problem size hash)
    cache: Mutex<HashMap<String, TunedConfig>>,
}

impl Default for Tuner {
    fn default() -> Self {
        Self::new()
    }
}

impl Tuner {
    /// Create a new Tuner instance.
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

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

        // 1. Check cache
        {
            let cache = self.cache.lock().unwrap();
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

        // 4. Update cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(key, best_config);
        }

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

        // unsafe {
        //    device.newBufferWithBytes_length_options(...)
        // }
        // The bindings might differ. Let's assume standard signatures.
        // If passing pointer directly, use unsafe block.
        // But obj2-metal wrapper might be safe if using &T?
        // Based on error "unsafe fn newBufferWithBytes...", we need unsafe.

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
}
