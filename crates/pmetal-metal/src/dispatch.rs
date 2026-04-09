//! Kernel dispatch router — selects Metal 3 or Metal 4 backend per operation.
//!
//! [`KernelDispatch`] is the single entry point for all kernel routing decisions.
//! It holds both backends and exposes methods that return the appropriate
//! [`KernelBackend`] reference for a given operation shape.
//!
//! # Routing policy
//!
//! - **`backend_for_gemm(m, n, k)`**: Returns Metal 4 when `#[cfg(has_metal4)]`
//!   and the Metal 4 backend is present and [`KernelBackend::should_handle_gemm`]
//!   returns `true`. Falls back to Metal 3 otherwise (decode M=1, unaligned K,
//!   or Metal 4 not compiled in).
//!
//! - **`preferred_backend()`**: Returns Metal 4 when available; Metal 3
//!   otherwise. Used for non-shape-dependent ops (e.g., MPP flash attention)
//!   where Metal 4 unconditionally wins.
//!
//! - **`metal3()`**: Explicit Metal 3 accessor for callers that need to bypass
//!   routing (e.g., fallback paths inside the Metal 4 backend itself).

use std::sync::Arc;

use crate::backend::KernelBackend;
use crate::context::MetalContext;
use crate::metal3_backend::Metal3Backend;
#[cfg(has_metal4)]
use crate::metal4::Metal4Backend;

// ============================================================================
// KernelDispatch
// ============================================================================

/// Routes kernel operations to the appropriate backend (Metal 3 or Metal 4).
///
/// On M5+ hardware built with the Metal 4 SDK, operations are routed to the
/// Metal 4 / MPP backend for supported shapes. All other operations fall back
/// to the Metal 3 backend.
///
/// Construct once per inference or training session via [`KernelDispatch::new`]
/// and share freely — `KernelDispatch` is `Send + Sync` because both backends are.
pub struct KernelDispatch {
    metal3: Metal3Backend,
    /// Metal 4 backend — present only on M5+ hardware with the Metal 4 SDK.
    ///
    /// Gated by `#[cfg(has_metal4)]` so the field does not exist at all on
    /// non-Metal-4 builds, keeping the struct layout identical to Metal 3 builds.
    #[cfg(has_metal4)]
    metal4: Option<Metal4Backend>,
}

impl KernelDispatch {
    /// Create a new dispatch router backed by the given Metal context.
    ///
    /// On Metal 4 builds, attempts to construct a [`Metal4Backend`] when the
    /// device has NAX cores and the Metal 4 library is loaded. Falls back to
    /// Metal 3-only routing if initialisation fails (missing SDK, non-M5
    /// hardware, or library load error).
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        let metal3 = Metal3Backend::new(ctx.clone());

        #[cfg(has_metal4)]
        let metal4 = if ctx.properties().has_nax
            && ctx.pipeline_cache().metal4_library().is_some()
        {
            match Metal4Backend::new(ctx.clone()) {
                Ok(m4) => {
                    tracing::info!("Metal 4 / MPP backend initialized");
                    Some(m4)
                }
                Err(e) => {
                    tracing::warn!("Metal 4 init failed, using Metal 3: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Self {
            metal3,
            #[cfg(has_metal4)]
            metal4,
        }
    }

    // ---- Routing methods ----------------------------------------------------

    /// Select the backend for a GEMM of the given shape.
    ///
    /// Returns Metal 4 when:
    /// - `has_metal4` is compiled in, **and**
    /// - `self.metal4` is `Some`, **and**
    /// - [`KernelBackend::should_handle_gemm`] on the Metal 4 backend returns `true`.
    ///
    /// Falls back to Metal 3 for:
    /// - Decode / matvec (M = 1) — MPP tile constraints waste threads.
    /// - K not divisible by 32 — NAX alignment requirement.
    /// - All non-Metal-4 builds.
    pub fn backend_for_gemm(&self, m: usize, n: usize, k: usize) -> &dyn KernelBackend {
        #[cfg(has_metal4)]
        if let Some(ref m4) = self.metal4 {
            if m4.should_handle_gemm(m, n, k) {
                return m4;
            }
        }
        // Suppress unused-variable warnings on Metal 3 builds.
        let _ = (m, n, k);
        &self.metal3
    }

    /// Select the preferred backend for non-shape-dependent operations.
    ///
    /// Returns Metal 4 when available (e.g., MPP flash attention); Metal 3
    /// otherwise. Operations that always require a specific backend (e.g.,
    /// training losses only on Metal 3) should call the backend directly via
    /// [`metal3`][KernelDispatch::metal3].
    pub fn preferred_backend(&self) -> &dyn KernelBackend {
        #[cfg(has_metal4)]
        if let Some(ref m4) = self.metal4 {
            return m4;
        }
        &self.metal3
    }

    /// Return the Metal 3 backend directly.
    ///
    /// Used for explicit fallback paths and for operations that Metal 4 does
    /// not support (training losses, etc.) as advertised by
    /// [`BackendCaps::metal4`][crate::backend::BackendCaps::metal4].
    ///
    /// Note: RoPE is wired to Metal 4 on M5+ hardware and is no longer a
    /// Metal-3-only operation.
    pub fn metal3(&self) -> &Metal3Backend {
        &self.metal3
    }
}
