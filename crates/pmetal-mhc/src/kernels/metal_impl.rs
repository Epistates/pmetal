//! Metal GPU implementation for mHC kernels.
//!
//! This module provides Rust bindings for the Metal compute shaders that accelerate
//! mHC operations on Apple Silicon.

use super::{KernelStats, MHC_METAL_SHADERS, MhcKernelConfig};
use crate::params::MhcMappings;
use crate::sinkhorn::SinkhornConfig;
use ndarray::{Array2, ArrayView1, ArrayView2};

#[cfg(feature = "metal")]
use objc2::rc::Retained;
#[cfg(feature = "metal")]
use objc2::runtime::ProtocolObject;
#[cfg(feature = "metal")]
use objc2_foundation::NSString;
#[cfg(feature = "metal")]
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
#[cfg(feature = "metal")]
use std::ptr::NonNull;

/// Metal context for mHC kernel execution.
#[cfg(feature = "metal")]
#[allow(dead_code)]
pub struct MhcMetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,

    // Compiled pipelines
    compute_mappings_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    apply_constraints_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    apply_pre_mapping_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    apply_post_res_mapping_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    sinkhorn_backward_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    compute_amax_gain_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    expand_to_streams_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    collapse_streams_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Configuration
    config: MhcKernelConfig,

    // Statistics
    stats: KernelStats,
}

#[cfg(feature = "metal")]
impl MhcMetalContext {
    /// Create a new Metal context for mHC operations.
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        config: MhcKernelConfig,
    ) -> Result<Self, MhcMetalError> {
        let queue = device
            .newCommandQueue()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        // Compile shader library from source
        let source = NSString::from_str(MHC_METAL_SHADERS);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
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
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MhcMetalError> {
        let ns_name = NSString::from_str(name);
        let function = library.newFunctionWithName(&ns_name).ok_or_else(|| {
            MhcMetalError::FunctionNotFound(name.to_string(), "not found in library".into())
        })?;

        device
            .newComputePipelineStateWithFunction_error(&function)
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
        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        // Stage 1: Compute mappings (flatten + RMSNorm + project)
        encoder.setComputePipelineState(&self.compute_mappings_pipeline);
        self.set_buffer(&encoder, 0, &alpha_pre_buf);
        self.set_buffer(&encoder, 1, &b_pre_buf);
        self.set_buffer(&encoder, 2, &h_pre_buf);
        self.set_buffer(&encoder, 3, &config_buf);

        let threads_per_group = MTLSize {
            width: self.config.compute_mappings_threads as usize,
            height: 1,
            depth: 1,
        };
        let num_groups = MTLSize {
            width: (n * n).div_ceil(threads_per_group.width),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threads_per_group);

        // Repeat for post and res (or use batch kernel)
        self.set_buffer(&encoder, 0, &alpha_post_buf);
        self.set_buffer(&encoder, 1, &b_post_buf);
        self.set_buffer(&encoder, 2, &h_post_buf);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threads_per_group);

        self.set_buffer(&encoder, 0, &alpha_res_buf);
        self.set_buffer(&encoder, 1, &b_res_buf);
        self.set_buffer(&encoder, 2, &h_res_buf);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threads_per_group);

        // Stage 2: Apply Sinkhorn constraints
        encoder.setComputePipelineState(&self.apply_constraints_pipeline);
        for buf in [&h_pre_buf, &h_post_buf, &h_res_buf] {
            self.set_buffer(&encoder, 0, buf);
            self.set_buffer(&encoder, 1, &config_buf);

            let sinkhorn_groups = MTLSize {
                width: n.div_ceil(self.config.sinkhorn_threads as usize),
                height: 1,
                depth: 1,
            };
            let sinkhorn_threads = MTLSize {
                width: self.config.sinkhorn_threads as usize,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(sinkhorn_groups, sinkhorn_threads);
        }

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

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

        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        encoder.setComputePipelineState(&self.apply_pre_mapping_pipeline);
        self.set_buffer(&encoder, 0, &h_pre_buf);
        self.set_buffer(&encoder, 1, &x_buf);
        self.set_buffer(&encoder, 2, &out_buf);
        self.set_buffer(&encoder, 3, &config_buf);

        let threads = MTLSize {
            width: self.config.apply_threads as usize,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: (n * c).div_ceil(threads.width),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

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

        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        encoder.setComputePipelineState(&self.apply_post_res_mapping_pipeline);
        self.set_buffer(&encoder, 0, &h_res_buf);
        self.set_buffer(&encoder, 1, &h_post_buf);
        self.set_buffer(&encoder, 2, &x_buf);
        self.set_buffer(&encoder, 3, &h_out_buf);
        self.set_buffer(&encoder, 4, &out_buf);
        self.set_buffer(&encoder, 5, &config_buf);

        let threads = MTLSize {
            width: self.config.apply_threads as usize,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: (n * c).div_ceil(threads.width),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

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

        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        encoder.setComputePipelineState(&self.expand_to_streams_pipeline);
        self.set_buffer(&encoder, 0, &x_buf);
        self.set_buffer(&encoder, 1, &out_buf);
        self.set_buffer(&encoder, 2, &config_buf);

        let threads = MTLSize {
            width: self.config.apply_threads as usize,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: (batch * n * c).div_ceil(threads.width),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

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

        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        encoder.setComputePipelineState(&self.collapse_streams_pipeline);
        self.set_buffer(&encoder, 0, &x_buf);
        self.set_buffer(&encoder, 1, &out_buf);
        self.set_buffer(&encoder, 2, &config_buf);

        let threads = MTLSize {
            width: self.config.apply_threads as usize,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: c.div_ceil(threads.width),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

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

        let cmd_buffer = self
            .queue
            .commandBuffer()
            .ok_or(MhcMetalError::DeviceNotFound)?;
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or(MhcMetalError::DeviceNotFound)?;

        encoder.setComputePipelineState(&self.compute_amax_gain_pipeline);
        self.set_buffer(&encoder, 0, &h_buf);
        self.set_buffer(&encoder, 1, &out_buf);
        self.set_buffer(&encoder, 2, &config_buf);

        let threads = MTLSize {
            width: self.config.sinkhorn_threads as usize,
            height: 1,
            depth: 1,
        };
        let unit = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(unit, threads);

        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

        let result = self.read_buffer::<f32>(&out_buf, 1)?;
        Ok(result[0])
    }

    // Helper methods for buffer management

    fn set_buffer(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: usize,
        buffer: &ProtocolObject<dyn MTLBuffer>,
    ) {
        // SAFETY: The buffer is a valid MTLBuffer created by our device, and the index
        // corresponds to a valid argument in the compute pipeline.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
        }
    }

    fn create_buffer_from_slice(
        &self,
        data: &[f32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MhcMetalError> {
        let size = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MhcMetalError::NonContiguousArray)?;
        // SAFETY: ptr is a valid non-null pointer from a live slice, and size matches.
        // Metal copies the data into the buffer.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MhcMetalError::DeviceNotFound)?;
        Ok(buffer)
    }

    fn create_buffer_from_array2(
        &self,
        data: &ArrayView2<f32>,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MhcMetalError> {
        let slice = data.as_slice().ok_or(MhcMetalError::NonContiguousArray)?;
        self.create_buffer_from_slice(slice)
    }

    fn create_buffer_from_struct<T>(
        &self,
        data: &T,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MhcMetalError> {
        let size = std::mem::size_of::<T>();
        let ptr = NonNull::new(data as *const T as *mut std::ffi::c_void)
            .ok_or(MhcMetalError::NonContiguousArray)?;
        // SAFETY: ptr is a valid non-null pointer to a T, and size matches.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MhcMetalError::DeviceNotFound)?;
        Ok(buffer)
    }

    fn create_empty_buffer<T>(
        &self,
        count: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MhcMetalError> {
        let size = count * std::mem::size_of::<T>();
        self.device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or(MhcMetalError::DeviceNotFound)
    }

    fn read_buffer<T: Clone>(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        count: usize,
    ) -> Result<Vec<T>, MhcMetalError> {
        let ptr = buffer.contents().as_ptr() as *const T;
        // SAFETY: the buffer was created with StorageModeShared and the GPU has completed,
        // so the pointer is valid for `count` elements of T.
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        Ok(slice.to_vec())
    }

    fn read_buffer_to_array2(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        rows: usize,
        cols: usize,
    ) -> Result<Array2<f32>, MhcMetalError> {
        let data = self.read_buffer::<f32>(buffer, rows * cols)?;
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| MhcMetalError::ShapeError(e.to_string()))
    }
}

// SAFETY: Metal protocol objects are thread-safe when used correctly.
// We ensure GPU work completes before reading back results.
#[cfg(feature = "metal")]
unsafe impl Send for MhcMetalContext {}
#[cfg(feature = "metal")]
unsafe impl Sync for MhcMetalContext {}

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
    let device = MTLCreateSystemDefaultDevice().ok_or(MhcMetalError::DeviceNotFound)?;
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
