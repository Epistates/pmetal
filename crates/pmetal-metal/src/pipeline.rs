//! Compute pipeline state caching.
//!
//! This module provides caching for Metal compute pipeline states,
//! which are expensive to create and should be reused.

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};
use std::collections::HashMap;
use std::ptr::NonNull;
use tracing::debug;

use crate::error::{MetalError, Result};

/// Typed function constant for Metal shader specialization.
#[derive(Debug, Clone, Copy)]
pub enum FunctionConstant {
    /// Boolean constant
    Bool(bool),
    /// Unsigned integer constant
    UInt(u32),
    /// Float constant
    Float(f32),
}

impl std::fmt::Display for FunctionConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionConstant::Bool(v) => write!(f, "{}", v),
            FunctionConstant::UInt(v) => write!(f, "{}", v),
            FunctionConstant::Float(v) => write!(f, "{}", v),
        }
    }
}

/// Cache for compute pipeline states.
///
/// Creating pipeline states is expensive as it involves shader compilation.
/// This cache stores compiled pipelines for reuse.
pub struct PipelineCache {
    /// Cached pipeline states, keyed by function name + config hash.
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,

    /// The Metal library containing our kernels.
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
}

impl PipelineCache {
    /// Create a new empty pipeline cache.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            library: None,
        }
    }

    /// Load the Metal library from embedded bytes.
    ///
    /// This should be called once during initialization.
    pub fn load_library(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        library_data: &[u8],
    ) -> Result<()> {
        // Create dispatch_data from the library bytes
        let dispatch_data = DispatchData::from_bytes(library_data);

        let library = device
            .newLibraryWithData_error(&dispatch_data)
            .map_err(|e| MetalError::LibraryLoad(e.to_string()))?;

        self.library = Some(library);
        debug!("Loaded Metal library ({} bytes)", library_data.len());

        Ok(())
    }

    /// Load the Metal library from source code.
    ///
    /// This compiles the shader at runtime (slower than pre-compiled).
    pub fn load_library_from_source(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
    ) -> Result<()> {
        let source_ns = NSString::from_str(source);

        let library = device
            .newLibraryWithSource_options_error(&source_ns, None)
            .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

        self.library = Some(library);
        debug!("Compiled Metal library from source");

        Ok(())
    }

    /// Get a reference to the loaded library.
    pub fn library(&self) -> Option<&ProtocolObject<dyn MTLLibrary>> {
        self.library.as_deref()
    }

    /// Get or create a pipeline state for a function.
    ///
    /// # Arguments
    ///
    /// * `device` - The Metal device
    /// * `function_name` - Name of the function in the library
    /// * `config_key` - Optional configuration key for variants
    ///
    /// # Returns
    ///
    /// A reference to the cached pipeline state.
    pub fn get_or_create_pipeline(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        function_name: &str,
        config_key: Option<&str>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let cache_key = match config_key {
            Some(key) => format!("{}:{}", function_name, key),
            None => function_name.to_string(),
        };

        // Check if already cached
        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return Ok(pipeline.clone());
        }

        // Get the library
        let library = self
            .library
            .as_ref()
            .ok_or_else(|| MetalError::LibraryLoad("Library not loaded".to_string()))?;

        // Get the function
        let function_ns = NSString::from_str(function_name);
        let function = library
            .newFunctionWithName(&function_ns)
            .ok_or_else(|| MetalError::FunctionNotFound(function_name.to_string()))?;

        // Create the pipeline state
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

        debug!(
            "Created pipeline for '{}' (max threads per threadgroup: {})",
            function_name,
            pipeline.maxTotalThreadsPerThreadgroup()
        );

        // Cache and return a clone
        self.pipelines.insert(cache_key, pipeline.clone());
        Ok(pipeline)
    }

    /// Get or create a pipeline state for a function with specialized constants.
    ///
    /// This allows creating specialized kernels (e.g., with specific tile sizes)
    /// using Metal function constants.
    pub fn get_or_create_specialized_pipeline(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        function_name: &str,
        constants: &HashMap<u64, u32>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Convert u32 constants to typed constants (all as UInt)
        let typed_constants: HashMap<u64, FunctionConstant> = constants
            .iter()
            .map(|(&k, &v)| (k, FunctionConstant::UInt(v)))
            .collect();
        self.get_or_create_specialized_pipeline_typed(device, function_name, &typed_constants)
    }

    /// Get or create a pipeline state for a function with typed specialized constants.
    ///
    /// This allows creating specialized kernels with different constant types
    /// (Bool, UInt, Float) using Metal function constants.
    pub fn get_or_create_specialized_pipeline_typed(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        function_name: &str,
        constants: &HashMap<u64, FunctionConstant>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Create a unique key for this configuration
        let mut sorted_keys: Vec<_> = constants.keys().collect();
        sorted_keys.sort();
        let config_str = sorted_keys
            .iter()
            .map(|&k| format!("{}={}", k, constants[k]))
            .collect::<Vec<_>>()
            .join(",");
        let cache_key = format!("{}:specialized:[{}]", function_name, config_str);

        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return Ok(pipeline.clone());
        }

        let library = self
            .library
            .as_ref()
            .ok_or_else(|| MetalError::LibraryLoad("Library not loaded".to_string()))?;

        let function_ns = NSString::from_str(function_name);

        // Create constant values object
        let constant_values = MTLFunctionConstantValues::new();
        for (&index, constant) in constants {
            unsafe {
                match constant {
                    FunctionConstant::Bool(value) => {
                        let ptr = value as *const bool as *mut std::ffi::c_void;
                        if let Some(non_null) = NonNull::new(ptr) {
                            constant_values.setConstantValue_type_atIndex(
                                non_null,
                                MTLDataType::Bool,
                                index as usize,
                            );
                        }
                    }
                    FunctionConstant::UInt(value) => {
                        let ptr = value as *const u32 as *mut std::ffi::c_void;
                        if let Some(non_null) = NonNull::new(ptr) {
                            constant_values.setConstantValue_type_atIndex(
                                non_null,
                                MTLDataType::UInt,
                                index as usize,
                            );
                        }
                    }
                    FunctionConstant::Float(value) => {
                        let ptr = value as *const f32 as *mut std::ffi::c_void;
                        if let Some(non_null) = NonNull::new(ptr) {
                            constant_values.setConstantValue_type_atIndex(
                                non_null,
                                MTLDataType::Float,
                                index as usize,
                            );
                        }
                    }
                }
            }
        }

        // Create specialized function
        let function = library
            .newFunctionWithName_constantValues_error(&function_ns, &constant_values)
            .map_err(|e| {
                MetalError::FunctionNotFound(format!("{} (specialized): {}", function_name, e))
            })?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

        debug!(
            "Created specialized pipeline for '{}' with config [{}] (max threads: {})",
            function_name,
            config_str,
            pipeline.maxTotalThreadsPerThreadgroup()
        );

        self.pipelines.insert(cache_key, pipeline.clone());
        Ok(pipeline)
    }

    /// Get pipeline thread execution width (SIMD width).
    ///
    /// This is typically 32 on Apple GPUs.
    pub fn get_thread_execution_width(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> u64 {
        pipeline.threadExecutionWidth() as u64
    }

    /// Get maximum threads per threadgroup for a pipeline.
    pub fn get_max_threads_per_threadgroup(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> u64 {
        pipeline.maxTotalThreadsPerThreadgroup() as u64
    }

    /// Clear all cached pipelines.
    ///
    /// This is useful for development when reloading shaders.
    pub fn clear(&mut self) {
        self.pipelines.clear();
        debug!("Cleared pipeline cache");
    }

    /// Get the number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PipelineCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineCache")
            .field("num_pipelines", &self.pipelines.len())
            .field("library_loaded", &self.library.is_some())
            .finish()
    }
}