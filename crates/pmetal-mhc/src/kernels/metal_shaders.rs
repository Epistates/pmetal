//! Metal shader source utilities.
//!
//! This module provides utilities for working with the embedded Metal shader source code.

use super::MHC_METAL_SHADERS;

/// Get the raw Metal shader source code.
pub fn shader_source() -> &'static str {
    MHC_METAL_SHADERS
}

/// Kernel names available in the shader library.
pub mod kernel_names {
    /// Fused RMSNorm + projection for computing mappings.
    pub const COMPUTE_MAPPINGS: &str = "compute_mappings";

    /// Apply sigmoid and Sinkhorn-Knopp constraints.
    pub const APPLY_CONSTRAINTS: &str = "apply_constraints";

    /// Apply pre-mapping: h_in = H^pre @ x
    pub const APPLY_PRE_MAPPING: &str = "apply_pre_mapping";

    /// Fused post-mapping and residual: x_{l+1} = H^res @ x + H^post^T @ h_out
    pub const APPLY_POST_RES_MAPPING: &str = "apply_post_res_mapping";

    /// Backward pass through Sinkhorn iterations.
    pub const SINKHORN_BACKWARD: &str = "sinkhorn_backward";

    /// Compute Amax gain magnitude for stability monitoring.
    pub const COMPUTE_AMAX_GAIN: &str = "compute_amax_gain";

    /// Expand single stream to n parallel streams.
    pub const EXPAND_TO_STREAMS: &str = "expand_to_streams";

    /// Collapse n streams to single stream via averaging.
    pub const COLLAPSE_STREAMS: &str = "collapse_streams";
}

/// Recommended thread group sizes for different kernels.
pub mod thread_groups {
    /// Thread group size for compute-heavy kernels (RMSNorm, projections).
    pub const COMPUTE_HEAVY: u32 = 256;

    /// Thread group size for Sinkhorn iterations (smaller for convergence).
    pub const SINKHORN: u32 = 64;

    /// Thread group size for matrix applications.
    pub const APPLY: u32 = 256;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_not_empty() {
        let source = shader_source();
        assert!(!source.is_empty());
        assert!(source.contains("compute_mappings"));
        assert!(source.contains("sinkhorn"));
    }

    #[test]
    fn test_kernel_names_defined() {
        // Just verify the constants are valid strings
        assert!(!kernel_names::COMPUTE_MAPPINGS.is_empty());
        assert!(!kernel_names::APPLY_CONSTRAINTS.is_empty());
        assert!(!kernel_names::SINKHORN_BACKWARD.is_empty());
    }
}
