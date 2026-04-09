//! Metal 4 command buffer wrapper with MPP dispatch support.
//!
//! Metal 4 replaces the `MTLCommandBuffer` / `MTLCommandQueue` model with a
//! new three-phase lifecycle:
//!
//! ```text
//! device.newCommandBuffer()
//!   └─► cb.beginCommandBufferWithAllocator(allocator)
//!         └─► encoder = cb.computeCommandEncoder()
//!               └─► ... encode work ...
//!               └─► encoder.endEncoding()
//!         └─► cb.endCommandBuffer()
//!   └─► queue.commit([cb], count: 1)
//! ```
//!
//! Skipping any step (or calling them out of order) is undefined behaviour
//! in Metal 4 and may corrupt GPU state or crash the process.  This module
//! enforces the correct ordering via an explicit state machine.
//!
//! # State machine
//!
//! ```text
//! Created ──begin()──► Began ──encoder()──► Encoding
//!                                               │
//!                                    end_and_commit() / Drop
//!                                               ▼
//!                                     endEncoding → endCommandBuffer
//!                                               │
//!                                       end_and_commit only
//!                                               ▼
//!                                          Committed
//! ```

#![allow(unsafe_code)]

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLDevice, MTLSharedEvent,
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4CommandQueue,
};
use tracing::trace;

use super::allocator_pool::CommandAllocatorPool;
use crate::error::{MetalError, Result};

// ============================================================================
// CbState — internal state machine
// ============================================================================

/// Lifecycle state for a [`Metal4CommandBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CbState {
    /// `newCommandBuffer` has been called; `beginCommandBuffer` has not.
    Created,
    /// `beginCommandBufferWithAllocator` has been called.
    Began,
    /// A compute encoder is active (`computeCommandEncoder` was called).
    Encoding,
    /// `endCommandBuffer` has been called and the buffer was submitted to the
    /// queue via `queue.commit`.
    Committed,
}

// ============================================================================
// Metal4CommandBuffer
// ============================================================================

/// A single Metal 4 command buffer ready for MPP / NAX kernel encoding.
///
/// Enforces the correct Metal 4 command buffer lifecycle via [`CbState`].
/// Holds pinned `Retained` references to resources accessed by the GPU so
/// that Arc/Retained reference counts keep them alive until this struct is
/// dropped (after GPU completion, in the typical usage pattern).
pub struct Metal4CommandBuffer {
    /// The underlying Metal 4 command buffer.
    cb: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    /// Active compute encoder, present only during `Encoding` state.
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    /// Pool that provided the allocator for this buffer.
    pool: Arc<CommandAllocatorPool>,
    /// Slot index in the pool (valid once `begin()` is called).
    pool_slot: Option<usize>,
    /// Current lifecycle state.
    state: CbState,
    /// Pinned resource references — keeps Metal buffers alive until this
    /// struct is dropped.
    _pinned: Vec<Retained<ProtocolObject<dyn MTLAllocation>>>,
}

impl Metal4CommandBuffer {
    /// Create a new command buffer from `device`.
    ///
    /// The buffer starts in [`CbState::Created`].  Call [`begin`][Self::begin]
    /// before encoding any work.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pool: Arc<CommandAllocatorPool>,
    ) -> Result<Self> {
        let cb = device.newCommandBuffer().ok_or(MetalError::CommandBufferCreation)?;
        Ok(Self {
            cb,
            encoder: None,
            pool,
            pool_slot: None,
            state: CbState::Created,
            _pinned: Vec::new(),
        })
    }

    /// Acquire an allocator from the pool and call `beginCommandBufferWithAllocator`.
    ///
    /// Transitions state from `Created` to `Began`.
    ///
    /// # Errors
    ///
    /// - `MetalError::InvalidConfig` if called from any state other than `Created`.
    /// - Propagates pool acquisition errors (GPU hang timeout).
    pub fn begin(&mut self) -> Result<()> {
        if self.state != CbState::Created {
            return Err(MetalError::InvalidConfig(format!(
                "Metal4CommandBuffer::begin called in {:?} state (expected Created)",
                self.state,
            )));
        }

        // Acquire an idle allocator and call beginCommandBufferWithAllocator.
        // `with_allocator` releases the pool lock before invoking the closure,
        // so the Obj-C call runs without holding the pool Mutex.
        let cb = &self.cb;
        let slot_idx = self.pool.with_allocator(|allocator| {
            cb.beginCommandBufferWithAllocator(allocator);
        })?;

        self.pool_slot = Some(slot_idx);
        self.state = CbState::Began;
        trace!("Metal4CommandBuffer: began (pool_slot={})", slot_idx);
        Ok(())
    }

    /// Create a compute command encoder.
    ///
    /// Transitions state from `Began` to `Encoding`.  The encoder reference is
    /// valid until [`end_and_commit`][Self::end_and_commit] or `Drop`.
    ///
    /// # Errors
    ///
    /// - `MetalError::InvalidConfig` if not in `Began` state.
    /// - `MetalError::EncoderCreation` if Metal fails to allocate the encoder.
    pub fn encoder(
        &mut self,
    ) -> Result<&ProtocolObject<dyn MTL4ComputeCommandEncoder>> {
        if self.state != CbState::Began {
            return Err(MetalError::InvalidConfig(format!(
                "Metal4CommandBuffer::encoder called in {:?} state (expected Began)",
                self.state,
            )));
        }

        let enc = self
            .cb
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        self.encoder = Some(enc);
        self.state = CbState::Encoding;
        trace!("Metal4CommandBuffer: compute encoder created");
        // SAFETY: encoder was just set to Some above.
        Ok(self.encoder.as_deref().unwrap())
    }

    /// Pin `resource` to prevent deallocation until this struct is dropped.
    ///
    /// Call this for every `MTLBuffer` or heap passed to the encoder to ensure
    /// it is not deallocated before the GPU finishes reading it.
    pub fn bind_resource(&mut self, resource: Retained<ProtocolObject<dyn MTLAllocation>>) {
        self._pinned.push(resource);
    }

    /// End encoding, end the command buffer, and commit it to `queue`.
    ///
    /// Transitions from `Encoding` to `Committed`.  Releases the allocator
    /// slot back to the pool, tagged with `(completion_event, completion_value)`
    /// so the pool can reclaim the allocator once the GPU signals completion.
    ///
    /// The caller is responsible for scheduling a `queue.signalEvent:value:`
    /// command that signals `completion_event` at `completion_value` on the
    /// GPU timeline so the pool knows when to call `allocator.reset()`.
    ///
    /// # Errors
    ///
    /// Returns `MetalError::InvalidConfig` if not in `Encoding` state.
    pub fn end_and_commit(
        &mut self,
        queue: &ProtocolObject<dyn MTL4CommandQueue>,
        completion_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        completion_value: u64,
    ) -> Result<()> {
        if self.state != CbState::Encoding {
            return Err(MetalError::InvalidConfig(format!(
                "Metal4CommandBuffer::end_and_commit called in {:?} state (expected Encoding)",
                self.state,
            )));
        }

        // End the active encoder.
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }

        // End the command buffer, freeing the allocator for reuse after GPU
        // completion (signalled via the event below).
        self.cb.endCommandBuffer();

        // Commit to the queue.
        // `commit_count` takes a pointer to an array of non-null command buffer
        // pointers.  We build a one-element stack array and pass a NonNull ptr.
        //
        // SAFETY:
        //   - `cb_ptr` is a valid NonNull pointing to a live MTL4CommandBuffer.
        //   - `array` is a one-element stack array with lifetime extending past
        //     the `commit_count` call.
        //   - `count = 1` matches the array length exactly.
        unsafe {
            let cb_ptr: NonNull<ProtocolObject<dyn MTL4CommandBuffer>> =
                NonNull::from(self.cb.as_ref());
            let mut array = [cb_ptr];
            queue.commit_count(
                NonNull::new_unchecked(array.as_mut_ptr()),
                1,
            );
        }

        // Release the allocator slot back to the pool.  The pool will call
        // `allocator.reset()` once `completion_event` reaches `completion_value`.
        if let Some(slot) = self.pool_slot.take() {
            self.pool.release(slot, completion_event, completion_value);
        }

        self.state = CbState::Committed;
        trace!("Metal4CommandBuffer: committed");
        Ok(())
    }

    /// Current lifecycle state (primarily for testing and assertions).
    pub fn state(&self) -> CbState {
        self.state
    }
}

// ============================================================================
// Drop — ensure correct teardown on early exit (e.g., error paths)
// ============================================================================

impl Drop for Metal4CommandBuffer {
    fn drop(&mut self) {
        // Unwind the state machine in reverse so Metal sees valid call order.
        match self.state {
            CbState::Encoding => {
                // End encoder first, then end the command buffer.
                if let Some(enc) = self.encoder.take() {
                    enc.endEncoding();
                }
                self.cb.endCommandBuffer();
                // No GPU work was submitted, so it's safe to immediately
                // reset the allocator rather than waiting for GPU completion.
                if let Some(slot) = self.pool_slot.take() {
                    self.pool.force_reset_slot(slot);
                }
            }
            CbState::Began => {
                // `beginCommandBuffer` was called but no encoder was opened.
                // We still need to end the command buffer.
                self.cb.endCommandBuffer();
                if let Some(slot) = self.pool_slot.take() {
                    self.pool.force_reset_slot(slot);
                }
            }
            CbState::Created | CbState::Committed => {
                // Created: nothing was started; nothing to unwind.
                // Committed: queue owns the buffer; nothing to unwind.
            }
        }
    }
}
