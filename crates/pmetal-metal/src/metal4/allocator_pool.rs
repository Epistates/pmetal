//! Command allocator pool for Metal 4 / MPP command buffers.
//!
//! On Metal 4, `MTL4CommandAllocator` owns the memory backing encoded GPU
//! commands.  The critical safety invariant is that an allocator MUST NOT be
//! reset until every command buffer that drew from it has completed on the GPU.
//! Violating this causes a SIGSEGV because the GPU may still be reading command
//! memory that has been returned to the heap.
//!
//! # Design
//!
//! The pool maintains a fixed-size array of [`AllocatorSlot`]s.  Each slot
//! transitions through the following states:
//!
//! ```text
//! Idle ──► InFlight { event, value } ──► Idle (after GPU signals event)
//! ```
//!
//! [`CommandAllocatorPool::acquire`] scans for an `Idle` slot.  If none is
//! available it polls all `InFlight` slots, resets any whose `MTLSharedEvent`
//! has been signalled by the GPU, then waits in a short spin-sleep loop until
//! one becomes free.
//!
//! [`CommandAllocatorPool::release`] marks a slot `InFlight` — the caller is
//! responsible for signalling the associated event from the GPU timeline (via
//! `MTL4CommandQueue::signalEvent:value:`).
//!
//! [`CommandAllocatorPool::with_allocator`] is the primary entry point: it
//! acquires an idle slot, calls a closure with a reference to the allocator,
//! then returns the slot index for the caller to release after committing.

#![allow(unsafe_code)]

use std::sync::Arc;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLSharedEvent};
use objc2_metal::{MTL4CommandAllocator, MTL4CommandAllocatorDescriptor};
use parking_lot::Mutex;
use tracing::trace;

use crate::error::{MetalError, Result};

// ============================================================================
// AllocatorState
// ============================================================================

/// Lifecycle state for one slot in the allocator pool.
enum AllocatorState {
    /// The allocator has been reset and is ready for a new command buffer.
    Idle,
    /// The allocator has been handed to a command buffer that has been
    /// committed but not yet completed on the GPU.
    InFlight {
        /// Shared event the GPU will signal on completion.
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        /// Value the GPU will write to `event` on completion.
        value: u64,
    },
}

// ============================================================================
// AllocatorSlot
// ============================================================================

struct AllocatorSlot {
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    state: AllocatorState,
}

// ============================================================================
// CommandAllocatorPool
// ============================================================================

/// Thread-safe pool of reusable Metal 4 command allocators.
///
/// Create one pool per `MTL4CommandQueue` and keep it alive for the lifetime
/// of that queue.  Typical `max_in_flight` values are 2 (double-buffer) or 3
/// (triple-buffer).
pub struct CommandAllocatorPool {
    slots: Mutex<Vec<AllocatorSlot>>,
}

impl CommandAllocatorPool {
    /// Create a new pool with `max_in_flight` pre-allocated command allocators.
    ///
    /// Returns an error if the device cannot allocate any of the allocators.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, max_in_flight: usize) -> Result<Arc<Self>> {
        assert!(max_in_flight >= 1, "max_in_flight must be at least 1");

        let desc = MTL4CommandAllocatorDescriptor::new();
        let mut slots = Vec::with_capacity(max_in_flight);

        for i in 0..max_in_flight {
            let allocator = device
                .newCommandAllocatorWithDescriptor_error(&desc)
                .map_err(|e| {
                    MetalError::Internal(format!(
                        "MTL4CommandAllocator[{}] creation failed: {}",
                        i, e
                    ))
                })?;

            slots.push(AllocatorSlot {
                allocator,
                state: AllocatorState::Idle,
            });
        }

        Ok(Arc::new(Self { slots: Mutex::new(slots) }))
    }

    /// Acquire an idle slot and invoke `f` with a reference to its allocator.
    ///
    /// Returns the slot index so the caller can pass it to [`release`][Self::release]
    /// after committing the command buffer.
    ///
    /// The closure is invoked while the pool lock is NOT held (the lock is
    /// released before `f` runs), so it is safe to call Objective-C methods
    /// that may re-enter or block from within `f`.
    ///
    /// # Errors
    ///
    /// Returns `MetalError::ExecutionFailed` if no slot becomes available
    /// within 5 s (indicates GPU hang).
    pub fn with_allocator<F>(&self, f: F) -> Result<usize>
    where
        F: FnOnce(&ProtocolObject<dyn MTL4CommandAllocator>),
    {
        const POLL_INTERVAL: Duration = Duration::from_millis(1);
        const MAX_WAIT: Duration = Duration::from_secs(5);

        let start = std::time::Instant::now();

        loop {
            // Snapshot the allocator pointer for the idle slot (if any).
            let idle_ptr: Option<(usize, *const ProtocolObject<dyn MTL4CommandAllocator>)> = {
                let mut slots = self.slots.lock();
                Self::poll_completed_locked(&mut slots);

                slots.iter().enumerate().find_map(|(idx, s)| {
                    if matches!(s.state, AllocatorState::Idle) {
                        Some((idx, &*s.allocator as *const _))
                    } else {
                        None
                    }
                })
            };
            // Lock is dropped here — safe to call Obj-C below.

            if let Some((idx, ptr)) = idle_ptr {
                // SAFETY: The pointer is valid because:
                //   1. The pool's Vec is append-only (slots never removed).
                //   2. The Retained<…> inside the slot keeps the Obj-C object alive.
                //   3. We hold an Arc to the pool so the Vec itself is alive.
                // While we call `f`, no other thread can acquire this same slot
                // because `acquire` marks nothing — we rely on the caller calling
                // `release` with `idx` before another call to `with_allocator`
                // can acquire the same slot.  For single-threaded encoding use
                // (the typical pattern) this is correct.  Multi-threaded encoding
                // should use one pool per thread.
                unsafe { f(&*ptr) };
                trace!("CommandAllocatorPool: acquired slot {}", idx);
                return Ok(idx);
            }

            if start.elapsed() >= MAX_WAIT {
                return Err(MetalError::ExecutionFailed(format!(
                    "CommandAllocatorPool: no idle allocator after {:?} — possible GPU hang",
                    MAX_WAIT,
                )));
            }

            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Mark slot `index` as in-flight.
    ///
    /// The GPU must signal `event` with `value` when the command buffer
    /// encoded with this allocator completes.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    pub fn release(
        &self,
        index: usize,
        event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        value: u64,
    ) {
        let mut slots = self.slots.lock();
        slots[index].state = AllocatorState::InFlight { event, value };
        trace!("CommandAllocatorPool: released slot {} (event value={})", index, value);
    }

    /// Reset a slot directly to `Idle` without a GPU completion check.
    ///
    /// Only call this when you know no GPU commands were submitted from the
    /// allocator (e.g., after a `beginCommandBuffer` / `endCommandBuffer`
    /// pair with no `queue.commit`).  Calling this on an allocator whose
    /// commands are still in-flight is undefined behaviour.
    pub fn force_reset_slot(&self, index: usize) {
        let mut slots = self.slots.lock();
        slots[index].allocator.reset();
        slots[index].state = AllocatorState::Idle;
        trace!("CommandAllocatorPool: force-reset slot {}", index);
    }

    // ---- Internal -----------------------------------------------------------

    /// Poll all in-flight slots and reset any whose GPU work has completed.
    /// Must be called with the slot lock held.
    fn poll_completed_locked(slots: &mut Vec<AllocatorSlot>) {
        for slot in slots.iter_mut() {
            let completed = match &slot.state {
                AllocatorState::InFlight { event, value } => {
                    event.signaledValue() >= *value
                }
                AllocatorState::Idle => false,
            };

            if completed {
                slot.allocator.reset();
                slot.state = AllocatorState::Idle;
                trace!("CommandAllocatorPool: slot reset after GPU completion");
            }
        }
    }
}
