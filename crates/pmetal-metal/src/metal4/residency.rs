//! Residency manager for Metal 4 / MPP weight tensors.
//!
//! Metal 4 introduces *explicit residency sets* (`MTLResidencySet`).  Every
//! buffer that GPU commands read from or write to MUST be registered in a
//! residency set that is attached to the command queue, or the GPU will see
//! unmapped memory and produce incorrect results (or crash).
//!
//! # Lifecycle
//!
//! ```text
//! ResidencyManager::new(device)
//!     └─► attach_to_queue(queue)   — wire set to command queue
//!         ├─► register(buf)        — add weight/activation buffers
//!         ├─► commit()             — batch-apply pending adds/removes
//!         └─► unregister(buf)      — mark for removal on next commit()
//! ```
//!
//! Weight buffers are registered once at model load time and kept resident
//! across all inference steps.  Activation buffers are registered before a
//! forward pass and unregistered after GPU completion.
//!
//! # Thread safety
//!
//! An `RwLock` guards the set.  Registration and unregistration acquire a
//! write lock; read-only queries (if any) use a read lock.  `commit()` takes
//! a write lock so that adds and removes are applied atomically.

#![allow(unsafe_code)]

use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
    MTL4CommandQueue,
};
use parking_lot::RwLock;
use tracing::trace;

use crate::error::{MetalError, Result};

// ============================================================================
// ResidencyManager
// ============================================================================

/// Manages a single `MTLResidencySet` for all buffers used by one command queue.
///
/// Attach to a queue with [`attach_to_queue`][Self::attach_to_queue], then
/// register every buffer that will be accessed by GPU commands.
pub struct ResidencyManager {
    inner: RwLock<ManagerInner>,
}

struct ManagerInner {
    /// The underlying Metal residency set.
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// Count of pending uncommitted registrations (for tracing).
    pending: usize,
}

impl ResidencyManager {
    /// Create a new residency manager for `device`.
    ///
    /// The residency set is created with an initial capacity of 64 slots, which
    /// covers a typical small model's weight buffers without reallocation.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Arc<Self>> {
        let desc = MTLResidencySetDescriptor::new();

        // SAFETY: `setInitialCapacity` is marked unsafe in the binding because
        // the Metal header doesn't bounds-check the value.  64 is a reasonable
        // initial hint and is not a hard limit.
        unsafe { desc.setInitialCapacity(64) };

        let set = device
            .newResidencySetWithDescriptor_error(&desc)
            .map_err(|e| {
                MetalError::Internal(format!("MTLResidencySet creation failed: {}", e))
            })?;

        Ok(Arc::new(Self {
            inner: RwLock::new(ManagerInner { set, pending: 0 }),
        }))
    }

    /// Attach this residency set to `queue` so that its resources are visible
    /// to every command buffer submitted on that queue.
    ///
    /// Must be called before any command buffers are committed.  Safe to call
    /// multiple times (Metal ignores redundant attachments).
    pub fn attach_to_queue(&self, queue: &ProtocolObject<dyn MTL4CommandQueue>) {
        let inner = self.inner.read();
        queue.addResidencySet(&inner.set);
        trace!("ResidencyManager: attached residency set to command queue");
    }

    /// Add `allocation` to the set.
    ///
    /// The addition is staged but not visible to the GPU until [`commit`][Self::commit]
    /// is called.
    pub fn register(&self, allocation: &ProtocolObject<dyn MTLAllocation>) {
        let mut inner = self.inner.write();
        inner.set.addAllocation(allocation);
        inner.pending += 1;
        trace!("ResidencyManager: registered allocation (pending={})", inner.pending);
    }

    /// Mark `allocation` for removal from the set.
    ///
    /// The removal is staged but not visible to the GPU until [`commit`][Self::commit]
    /// is called.
    pub fn unregister(&self, allocation: &ProtocolObject<dyn MTLAllocation>) {
        let mut inner = self.inner.write();
        inner.set.removeAllocation(allocation);
        inner.pending += 1;
        trace!("ResidencyManager: unregistered allocation (pending={})", inner.pending);
    }

    /// Commit all pending adds and removes to the residency set.
    ///
    /// After this call returns, newly registered resources will be made
    /// resident before the next GPU dispatch, and unregistered resources will
    /// be released from residency.
    ///
    /// Call this after bulk-registering model weights (once at load time) and
    /// after any per-step activation changes.
    pub fn commit(&self) {
        let mut inner = self.inner.write();
        inner.set.commit();
        trace!("ResidencyManager: committed {} pending changes", inner.pending);
        inner.pending = 0;
    }

    /// Return the current number of unique allocations in the set (including
    /// uncommitted additions).
    pub fn allocation_count(&self) -> usize {
        let inner = self.inner.read();
        inner.set.allocationCount() as usize
    }
}
