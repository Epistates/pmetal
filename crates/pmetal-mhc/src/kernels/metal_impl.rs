//! Metal GPU implementation for mHC kernels.
//!
//! This module provides Rust bindings for the Metal compute shaders that accelerate
//! mHC operations on Apple Silicon.

use super::{KernelStats, MhcKernelConfig, MHC_METAL_SHADERS};
use crate::params::MhcMappings;
use crate::sinkhorn::SinkhornConfig;
use ndarray::{Array2, ArrayView1, ArrayView2};

#[cfg(feature = "metal")]
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

/// Metal context for mHC kernel execution.
#[cfg(feature = "metal")]
#[allow(dead_code)]
pub struct MhcMetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,

    // Compiled pipelines
    compute_mappings_pipeline: ComputePipelineState,
    apply_constraints_pipeline: ComputePipelineState,
    apply_pre_mapping_pipeline: ComputePipelineState,
    apply_post_res_mapping_pipeline: ComputePipelineState,
    sinkhorn_backward_pipeline: ComputePipelineState,
    compute_amax_gain_pipeline: ComputePipelineState,
    expand_to_streams_pipeline: ComputePipelineState,
    collapse_streams_pipeline: ComputePipelineState,

    // Configuration
    config: MhcKernelConfig,

    // Statistics
    stats: KernelStats,
}

#[cfg(feature = "metal")]
impl MhcMetalContext {
    /// Create a new Metal context for mHC operations.
    pub fn new(device: Device, config: MhcKernelConfig) -> Result<Self, MhcMetalError> {
        let queue = device.new_command_queue();

        // Compile shader library
        let library = device
            .new_library_with_source(MHC_METAL_SHADERS, &metal::CompileOptions::new())
            .map_err(|e| MhcMetalError::CompileError(e.to_string()))?;

        // Create compute pipelines
        let compute_mappings_pipeline =
            Self::create_pipeline(&device, &library, "compute_mappings")?;
        let apply_constraints_pipeline =
            Self::create_pipeline(&device, &library, "apply_constraints")?;
        let apply_pre_mapping_pipeline =
            Self::create_pipeline(&device, &library, "apply_pre_mapping")?;
        let apply_post_res_mapping_pipeline =
            Self::create_pipeline(&device, &library, "apply_post_res_mapping")?;
        let sinkhorn_backward_pipeline =
            Self::create_pipeline(&device, &library, "sinkhorn_backward")?;
        let compute_amax_gain_pipeline =
            Self::create_pipeline(&device, &library, "compute_amax_gain")?;
        let expand_to_streams_pipeline =
            Self::create_pipeline(&device, &library, "expand_to_streams")?;
        let collapse_streams_pipeline =
            Self::create_pipeline(&device, &library, "collapse_streams")?;

        Ok(Self {
            device,
            queue,
            library,
            compute_mappings_pipeline,
            apply_constraints_pipeline,
            apply_pre_mapping_pipeline,
            apply_post_res_mapping_pipeline,
            sinkhorn_backward_pipeline,
            compute_amax_gain_pipeline,
            expand_to_streams_pipeline,
            collapse_streams_pipeline,
            config,
            stats: KernelStats::default(),
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, MhcMetalError> {
        let function = library
            .get_function(name, None)
            .map_err(|e| MhcMetalError::FunctionNotFound(name.to_string(), e.to_string()))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MhcMetalError::PipelineError(name.to_string(), e.to_string()))
    }

    /// Get accumulated kernel statistics.
    pub fn stats(&self) -> &KernelStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = KernelStats::default();
    }

    /// Compute mHC mappings on GPU (fused RMSNorm + projection + Sinkhorn).
    pub fn compute_mappings(
        &mut self,
        alpha_pre: ArrayView1<f32>,
        alpha_post: ArrayView1<f32>,
        alpha_res: ArrayView1<f32>,
        b_pre: ArrayView1<f32>,
        b_post: ArrayView1<f32>,
        b_res: ArrayView1<f32>,
        _phi_pre: ArrayView2<f32>,
        _phi_post: ArrayView2<f32>,
        _phi_res: ArrayView2<f32>,
        _rmsnorm_weight: ArrayView1<f32>,
        sinkhorn_config: &SinkhornConfig,
    ) -> Result<MhcMappings, MhcMetalError> {
        let start = std::time::Instant::now();
        let n = (alpha_pre.len() as f32).sqrt() as usize;

        // Create buffers
        let alpha_pre_buf = self.create_buffer_from_slice(alpha_pre.as_slice().unwrap())?;
        let alpha_post_buf = self.create_buffer_from_slice(alpha_post.as_slice().unwrap())?;
        let alpha_res_buf = self.create_buffer_from_slice(alpha_res.as_slice().unwrap())?;
        let b_pre_buf = self.create_buffer_from_slice(b_pre.as_slice().unwrap())?;
        let b_post_buf = self.create_buffer_from_slice(b_post.as_slice().unwrap())?;
        let b_res_buf = self.create_buffer_from_slice(b_res.as_slice().unwrap())?;

        // Output buffers
        let h_pre_buf = self.create_empty_buffer::<f32>(n * n)?;
        let h_post_buf = self.create_empty_buffer::<f32>(n * n)?;
        let h_res_buf = self.create_empty_buffer::<f32>(n * n)?;

        // Config buffer
        let config_data = MappingsConfig {
            n: n as u32,
            sinkhorn_iterations: sinkhorn_config.max_iterations as u32,
            epsilon: sinkhorn_config.epsilon,
            _padding: 0,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        // Create command buffer
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Stage 1: Compute mappings (flatten + RMSNorm + project)
        encoder.set_compute_pipeline_state(&self.compute_mappings_pipeline);
        encoder.set_buffer(0, Some(&alpha_pre_buf), 0);
        encoder.set_buffer(1, Some(&b_pre_buf), 0);
        encoder.set_buffer(2, Some(&h_pre_buf), 0);
        encoder.set_buffer(3, Some(&config_buf), 0);

        let threads_per_group = MTLSize::new(self.config.compute_mappings_threads as u64, 1, 1);
        let num_groups = MTLSize::new(
            ((n * n) as u64).div_ceil(threads_per_group.width),
            1,
            1,
        );
        encoder.dispatch_thread_groups(num_groups, threads_per_group);

        // Repeat for post and res (or use batch kernel)
        encoder.set_buffer(0, Some(&alpha_post_buf), 0);
        encoder.set_buffer(1, Some(&b_post_buf), 0);
        encoder.set_buffer(2, Some(&h_post_buf), 0);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);

        encoder.set_buffer(0, Some(&alpha_res_buf), 0);
        encoder.set_buffer(1, Some(&b_res_buf), 0);
        encoder.set_buffer(2, Some(&h_res_buf), 0);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);

        // Stage 2: Apply Sinkhorn constraints
        encoder.set_compute_pipeline_state(&self.apply_constraints_pipeline);
        for buf in [&h_pre_buf, &h_post_buf, &h_res_buf] {
            encoder.set_buffer(0, Some(buf), 0);
            encoder.set_buffer(1, Some(&config_buf), 0);

            let sinkhorn_groups = MTLSize::new(
                (n as u64).div_ceil(self.config.sinkhorn_threads as u64),
                1,
                1,
            );
            let sinkhorn_threads = MTLSize::new(self.config.sinkhorn_threads as u64, 1, 1);
            encoder.dispatch_thread_groups(sinkhorn_groups, sinkhorn_threads);
        }

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Read back results - reshape to include batch dimension (batch=1)
        let h_pre_raw = self.read_buffer::<f32>(&h_pre_buf, n * n)?;
        let h_post_raw = self.read_buffer::<f32>(&h_post_buf, n * n)?;
        let h_res_raw = self.read_buffer::<f32>(&h_res_buf, n * n)?;

        // MhcMappings expects: h_pre [batch, n], h_post [batch, n], h_res [batch, n, n]
        // For single computation, batch=1
        let h_pre = Array2::from_shape_vec((1, n), h_pre_raw[..n].to_vec())
            .map_err(|e| MhcMetalError::ShapeError(e.to_string()))?;
        let h_post = Array2::from_shape_vec((1, n), h_post_raw[..n].to_vec())
            .map_err(|e| MhcMetalError::ShapeError(e.to_string()))?;
        let h_res = ndarray::Array3::from_shape_vec((1, n, n), h_res_raw)
            .map_err(|e| MhcMetalError::ShapeError(e.to_string()))?;

        self.stats.compute_mappings_us += start.elapsed().as_micros() as u64;
        self.stats.invocations += 1;

        Ok(MhcMappings {
            h_pre,
            h_post,
            h_res,
        })
    }

    /// Apply pre-mapping on GPU: h_in = H^pre @ x
    pub fn apply_pre_mapping(
        &mut self,
        h_pre: ArrayView2<f32>,
        x: ArrayView2<f32>,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let start = std::time::Instant::now();
        let (n, c) = x.dim();

        let h_pre_buf = self.create_buffer_from_array2(&h_pre)?;
        let x_buf = self.create_buffer_from_array2(&x)?;
        let out_buf = self.create_empty_buffer::<f32>(n * c)?;

        let config_data = ApplyConfig {
            n: n as u32,
            c: c as u32,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.apply_pre_mapping_pipeline);
        encoder.set_buffer(0, Some(&h_pre_buf), 0);
        encoder.set_buffer(1, Some(&x_buf), 0);
        encoder.set_buffer(2, Some(&out_buf), 0);
        encoder.set_buffer(3, Some(&config_buf), 0);

        let threads = MTLSize::new(self.config.apply_threads as u64, 1, 1);
        let groups = MTLSize::new(((n * c) as u64).div_ceil(threads.width), 1, 1);
        encoder.dispatch_thread_groups(groups, threads);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let result = self.read_buffer_to_array2(&out_buf, n, c)?;

        self.stats.apply_us += start.elapsed().as_micros() as u64;
        self.stats.invocations += 1;

        Ok(result)
    }

    /// Apply fused post-mapping and residual: x_{l+1} = H^res @ x + H^post^T @ h_out
    pub fn apply_post_res_mapping(
        &mut self,
        h_res: ArrayView2<f32>,
        h_post: ArrayView2<f32>,
        x: ArrayView2<f32>,
        h_out: ArrayView2<f32>,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let start = std::time::Instant::now();
        let (n, c) = x.dim();

        let h_res_buf = self.create_buffer_from_array2(&h_res)?;
        let h_post_buf = self.create_buffer_from_array2(&h_post)?;
        let x_buf = self.create_buffer_from_array2(&x)?;
        let h_out_buf = self.create_buffer_from_array2(&h_out)?;
        let out_buf = self.create_empty_buffer::<f32>(n * c)?;

        let config_data = ApplyConfig {
            n: n as u32,
            c: c as u32,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.apply_post_res_mapping_pipeline);
        encoder.set_buffer(0, Some(&h_res_buf), 0);
        encoder.set_buffer(1, Some(&h_post_buf), 0);
        encoder.set_buffer(2, Some(&x_buf), 0);
        encoder.set_buffer(3, Some(&h_out_buf), 0);
        encoder.set_buffer(4, Some(&out_buf), 0);
        encoder.set_buffer(5, Some(&config_buf), 0);

        let threads = MTLSize::new(self.config.apply_threads as u64, 1, 1);
        let groups = MTLSize::new(((n * c) as u64).div_ceil(threads.width), 1, 1);
        encoder.dispatch_thread_groups(groups, threads);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let result = self.read_buffer_to_array2(&out_buf, n, c)?;

        self.stats.apply_us += start.elapsed().as_micros() as u64;
        self.stats.invocations += 1;

        Ok(result)
    }

    /// Expand single stream to n streams on GPU.
    pub fn expand_to_streams(
        &mut self,
        x: ArrayView2<f32>,
        n: usize,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let (batch, c) = x.dim();

        let x_buf = self.create_buffer_from_array2(&x)?;
        let out_buf = self.create_empty_buffer::<f32>(batch * n * c)?;

        let config_data = ExpandConfig {
            batch: batch as u32,
            n: n as u32,
            c: c as u32,
            _padding: 0,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.expand_to_streams_pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&config_buf), 0);

        let threads = MTLSize::new(self.config.apply_threads as u64, 1, 1);
        let groups = MTLSize::new(
            ((batch * n * c) as u64).div_ceil(threads.width),
            1,
            1,
        );
        encoder.dispatch_thread_groups(groups, threads);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        self.read_buffer_to_array2(&out_buf, n, c)
    }

    /// Collapse n streams to single stream (mean) on GPU.
    pub fn collapse_streams(
        &mut self,
        x: ArrayView2<f32>,
        n: usize,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let (n_streams, c) = x.dim();
        assert_eq!(n_streams, n, "Expected {} streams, got {}", n, n_streams);

        let x_buf = self.create_buffer_from_array2(&x)?;
        let out_buf = self.create_empty_buffer::<f32>(c)?;

        let config_data = ExpandConfig {
            batch: 1,
            n: n as u32,
            c: c as u32,
            _padding: 0,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.collapse_streams_pipeline);
        encoder.set_buffer(0, Some(&x_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&config_buf), 0);

        let threads = MTLSize::new(self.config.apply_threads as u64, 1, 1);
        let groups = MTLSize::new((c as u64).div_ceil(threads.width), 1, 1);
        encoder.dispatch_thread_groups(groups, threads);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        self.read_buffer_to_array2(&out_buf, 1, c)
    }

    /// Compute Amax gain magnitude on GPU.
    pub fn compute_amax_gain(&mut self, h: ArrayView2<f32>) -> Result<f32, MhcMetalError> {
        let (n, _) = h.dim();

        let h_buf = self.create_buffer_from_array2(&h)?;
        let out_buf = self.create_empty_buffer::<f32>(1)?;

        let config_data = ApplyConfig {
            n: n as u32,
            c: n as u32,
        };
        let config_buf = self.create_buffer_from_struct(&config_data)?;

        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.compute_amax_gain_pipeline);
        encoder.set_buffer(0, Some(&h_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.set_buffer(2, Some(&config_buf), 0);

        let threads = MTLSize::new(self.config.sinkhorn_threads as u64, 1, 1);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), threads);

        encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        let result = self.read_buffer::<f32>(&out_buf, 1)?;
        Ok(result[0])
    }

    // Helper methods for buffer management

    fn create_buffer_from_slice(&self, data: &[f32]) -> Result<Buffer, MhcMetalError> {
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    fn create_buffer_from_array2(&self, data: &ArrayView2<f32>) -> Result<Buffer, MhcMetalError> {
        let slice = data.as_slice().ok_or(MhcMetalError::NonContiguousArray)?;
        self.create_buffer_from_slice(slice)
    }

    fn create_buffer_from_struct<T>(&self, data: &T) -> Result<Buffer, MhcMetalError> {
        let buffer = self.device.new_buffer_with_data(
            data as *const T as *const _,
            std::mem::size_of::<T>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    fn create_empty_buffer<T>(&self, count: usize) -> Result<Buffer, MhcMetalError> {
        let buffer = self.device.new_buffer(
            (count * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    fn read_buffer<T: Clone>(
        &self,
        buffer: &Buffer,
        count: usize,
    ) -> Result<Vec<T>, MhcMetalError> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    fn read_buffer_to_array2(
        &self,
        buffer: &Buffer,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let data = self.read_buffer::<f32>(buffer, rows * cols)?;
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| MhcMetalError::ShapeError(e.to_string()))
    }
}

/// Configuration struct for mappings kernel.
#[repr(C)]
#[derive(Clone, Copy)]
struct MappingsConfig {
    n: u32,
    sinkhorn_iterations: u32,
    epsilon: f32,
    _padding: u32,
}

/// Configuration struct for apply kernels.
#[repr(C)]
#[derive(Clone, Copy)]
struct ApplyConfig {
    n: u32,
    c: u32,
}

/// Configuration struct for expand/collapse kernels.
#[repr(C)]
#[derive(Clone, Copy)]
struct ExpandConfig {
    batch: u32,
    n: u32,
    c: u32,
    _padding: u32,
}

/// Errors from Metal operations.
#[derive(Debug, Clone)]
pub enum MhcMetalError {
    /// Shader compilation failed.
    CompileError(String),
    /// Kernel function not found.
    FunctionNotFound(String, String),
    /// Pipeline creation failed.
    PipelineError(String, String),
    /// Array is not contiguous in memory.
    NonContiguousArray,
    /// Shape mismatch.
    ShapeError(String),
    /// Metal device not available.
    DeviceNotFound,
}

impl std::fmt::Display for MhcMetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MhcMetalError::CompileError(e) => write!(f, "Shader compilation error: {}", e),
            MhcMetalError::FunctionNotFound(name, e) => {
                write!(f, "Kernel function '{}' not found: {}", name, e)
            }
            MhcMetalError::PipelineError(name, e) => {
                write!(f, "Pipeline creation failed for '{}': {}", name, e)
            }
            MhcMetalError::NonContiguousArray => write!(f, "Array is not contiguous in memory"),
            MhcMetalError::ShapeError(e) => write!(f, "Shape error: {}", e),
            MhcMetalError::DeviceNotFound => write!(f, "Metal device not available"),
        }
    }
}

impl std::error::Error for MhcMetalError {}

/// Create a Metal context using the system default device.
#[cfg(feature = "metal")]
pub fn create_default_context(config: MhcKernelConfig) -> Result<MhcMetalContext, MhcMetalError> {
    let device = Device::system_default().ok_or(MhcMetalError::DeviceNotFound)?;
    MhcMetalContext::new(device, config)
}

// Fallback implementations when Metal is not available
#[cfg(not(feature = "metal"))]
pub struct MhcMetalContext;

#[cfg(not(feature = "metal"))]
impl MhcMetalContext {
    pub fn new(_config: MhcKernelConfig) -> Result<Self, MhcMetalError> {
        Err(MhcMetalError::DeviceNotFound)
    }
}

#[cfg(not(feature = "metal"))]
#[derive(Debug, Clone)]
pub enum MhcMetalError {
    DeviceNotFound,
}

#[cfg(not(feature = "metal"))]
impl std::fmt::Display for MhcMetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metal is not available on this platform")
    }
}

#[cfg(not(feature = "metal"))]
impl std::error::Error for MhcMetalError {}
