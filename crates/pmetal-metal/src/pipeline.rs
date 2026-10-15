#![allow(unsafe_code)]

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

impl FunctionConstant {
    /// A string that names the constant's type and exact value.
    ///
    /// `UInt(1)` and `Float(1.0)` display alike but reach the shader as
    /// different bytes, and floats are keyed by bit pattern so `0.0` and
    /// `-0.0` stay apart.
    fn key_fragment(&self) -> String {
        match self {
            FunctionConstant::Bool(v) => format!("bool:{v}"),
            FunctionConstant::UInt(v) => format!("uint:{v}"),
            FunctionConstant::Float(v) => format!("float:{:08x}", v.to_bits()),
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

    /// The Metal 3 library containing standard kernels.
    library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,

    /// The Metal 4 / MPP library containing NAX-accelerated kernels (M5+).
    /// Only loaded when has_metal4 cfg is set and the GPU supports NAX.
    metal4_library: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
}

impl PipelineCache {
    /// Create a new empty pipeline cache.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            library: None,
            metal4_library: None,
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

    /// Load the Metal 4 / MPP library from embedded bytes.
    ///
    /// This library contains NAX-accelerated kernels for M5+ GPUs.
    /// Only call when the device supports NAX (has_nax = true).
    pub fn load_metal4_library(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        library_data: &[u8],
    ) -> Result<()> {
        let dispatch_data = DispatchData::from_bytes(library_data);

        let library = device
            .newLibraryWithData_error(&dispatch_data)
            .map_err(|e| MetalError::LibraryLoad(format!("Metal 4 library: {}", e)))?;

        self.metal4_library = Some(library);
        debug!(
            "Loaded Metal 4 / MPP library ({} bytes)",
            library_data.len()
        );

        Ok(())
    }

    /// Get a reference to the loaded library.
    pub fn library(&self) -> Option<&ProtocolObject<dyn MTLLibrary>> {
        self.library.as_deref()
    }

    /// Get a reference to the Metal 4 / MPP library.
    pub fn metal4_library(&self) -> Option<&ProtocolObject<dyn MTLLibrary>> {
        self.metal4_library.as_deref()
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

    /// Get or create a pipeline from the Metal 4 / MPP library.
    ///
    /// Looks up the function in the Metal 4 library. Returns an error if the
    /// Metal 4 library is not loaded (caller should check `has_nax` first).
    pub fn get_or_create_metal4_pipeline(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        function_name: &str,
        constants: &HashMap<u64, FunctionConstant>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        // Prefixed so a Metal 4 entry can never alias a Metal 3 one.
        let cache_key = format!("metal4:{}", specialized_key(function_name, constants));
        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return Ok(pipeline.clone());
        }

        let library = self.metal4_library.as_ref().ok_or_else(|| {
            MetalError::LibraryLoad("Metal 4 / MPP library not loaded".to_string())
        })?;
        let pipeline = specialized_pipeline(library, device, function_name, constants)?;
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
        let cache_key = specialized_key(function_name, constants);
        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return Ok(pipeline.clone());
        }

        let library = self
            .library
            .as_ref()
            .ok_or_else(|| MetalError::LibraryLoad("Library not loaded".to_string()))?;
        let pipeline = specialized_pipeline(library, device, function_name, constants)?;
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

/// The cache key of a function specialized with `constants`.
///
/// Two requests share a key only if they agree on every constant's index, type
/// and exact value.
fn specialized_key(function_name: &str, constants: &HashMap<u64, FunctionConstant>) -> String {
    let mut indices: Vec<u64> = constants.keys().copied().collect();
    indices.sort_unstable();
    let list = indices
        .iter()
        .map(|index| format!("{index}={}", constants[index].key_fragment()))
        .collect::<Vec<_>>()
        .join(",");
    format!("{function_name}:specialized:[{list}]")
}

/// Build `function_name` from `library` with `constants` and compile it.
fn specialized_pipeline(
    library: &ProtocolObject<dyn MTLLibrary>,
    device: &ProtocolObject<dyn MTLDevice>,
    function_name: &str,
    constants: &HashMap<u64, FunctionConstant>,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let function_ns = NSString::from_str(function_name);

    let constant_values = MTLFunctionConstantValues::new();
    for (&index, constant) in constants {
        let (ptr, data_type) = match constant {
            FunctionConstant::Bool(value) => (NonNull::from(value).cast(), MTLDataType::Bool),
            FunctionConstant::UInt(value) => (NonNull::from(value).cast(), MTLDataType::UInt),
            FunctionConstant::Float(value) => (NonNull::from(value).cast(), MTLDataType::Float),
        };
        // SAFETY: `ptr` points to a live `bool`, `u32` or `f32` matching
        // `data_type`; Metal copies the value during the call.
        unsafe {
            constant_values.setConstantValue_type_atIndex(ptr, data_type, index as usize);
        }
    }

    let function = library
        .newFunctionWithName_constantValues_error(&function_ns, &constant_values)
        .map_err(|e| MetalError::FunctionNotFound(format!("{function_name} (specialized): {e}")))?;

    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

    debug!(
        "Created specialized pipeline for '{}' with {} constants (max threads: {})",
        function_name,
        constants.len(),
        pipeline.maxTotalThreadsPerThreadgroup()
    );
    Ok(pipeline)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MetalContext;

    fn constants(pairs: &[(u64, FunctionConstant)]) -> HashMap<u64, FunctionConstant> {
        pairs.iter().copied().collect()
    }

    /// Constants that print the same but reach the shader differently must
    /// not share a pipeline.
    #[test]
    fn keys_tell_constant_types_apart() {
        let key = |c: FunctionConstant| specialized_key("f", &constants(&[(0, c)]));
        assert_ne!(
            key(FunctionConstant::UInt(1)),
            key(FunctionConstant::Float(1.0))
        );
        assert_ne!(
            key(FunctionConstant::UInt(1)),
            key(FunctionConstant::Bool(true))
        );
        assert_ne!(
            key(FunctionConstant::Float(0.0)),
            key(FunctionConstant::Float(-0.0))
        );
        assert_ne!(
            key(FunctionConstant::Float(0.1)),
            key(FunctionConstant::Float(0.1 + f32::EPSILON))
        );
        assert_eq!(
            key(FunctionConstant::UInt(7)),
            key(FunctionConstant::UInt(7))
        );
    }

    #[test]
    fn keys_tell_indices_and_functions_apart_and_ignore_insertion_order() {
        let a = constants(&[
            (0, FunctionConstant::UInt(1)),
            (1, FunctionConstant::UInt(2)),
        ]);
        let b = constants(&[
            (1, FunctionConstant::UInt(2)),
            (0, FunctionConstant::UInt(1)),
        ]);
        assert_eq!(specialized_key("f", &a), specialized_key("f", &b));
        assert_ne!(specialized_key("f", &a), specialized_key("g", &a));
        // The same values at other indices are another specialization.
        let swapped = constants(&[
            (0, FunctionConstant::UInt(2)),
            (1, FunctionConstant::UInt(1)),
        ]);
        assert_ne!(specialized_key("f", &a), specialized_key("f", &swapped));
    }

    /// Two specializations of one kernel are two pipelines, and asking again
    /// gets the cached one back.
    #[test]
    fn specialized_pipelines_are_cached_per_constant_set() {
        let ctx = MetalContext::new().unwrap();
        let mut cache = ctx.pipeline_cache_mut();
        let before = cache.len();
        let with = |threads: u32| constants(&[(0, FunctionConstant::UInt(threads))]);

        let small = cache
            .get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "fused_linear_cross_entropy_forward",
                &with(32),
            )
            .unwrap();
        let large = cache
            .get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "fused_linear_cross_entropy_forward",
                &with(256),
            )
            .unwrap();
        let small_again = cache
            .get_or_create_specialized_pipeline_typed(
                ctx.device(),
                "fused_linear_cross_entropy_forward",
                &with(32),
            )
            .unwrap();

        assert_eq!(cache.len(), before + 2);
        assert!(!std::ptr::eq(&*small, &*large));
        assert!(std::ptr::eq(&*small, &*small_again));
    }
}
