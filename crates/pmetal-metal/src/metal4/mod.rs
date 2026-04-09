//! Metal 4 / MPP backend for Apple M5+ (Apple10, NAX cores).
//!
//! Compiled only when `cfg(has_metal4)` is set (macOS 26.0+ SDK, Metal >= 4.0).
//! Activates at runtime only when `has_nax` is true (M5+ hardware).
//!
//! # Structure
//!
//! - [`backend`] — [`Metal4Backend`] implementing [`KernelBackend`]. Currently
//!   delegates all operations to [`Metal3Backend`]; Tasks 11–18 replace calls
//!   with MPP kernel dispatch.
//!
//! - [`allocator_pool`] — Reusable `MTLCommandAllocator` pool (Task 7).
//!
//! - [`residency`] — Explicit residency sets for model weight buffers (Task 8).
//!
//! - [`command_buffer`] — Metal 4 command buffer wrapper for MPP encoding (Task 9).
//!
//! [`Metal3Backend`]: crate::metal3_backend::Metal3Backend

pub mod allocator_pool;
pub mod command_buffer;
pub mod residency;

mod backend;
pub use backend::Metal4Backend;
