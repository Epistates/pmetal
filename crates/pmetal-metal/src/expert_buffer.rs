#![allow(unsafe_code)]

//! 2 MB-aligned buffer pool for expert weight offloading.
//!
//! Expert weight offloading streams quantized MoE expert weights from disk into
//! GPU-visible memory at decode time, hiding I/O latency behind compute.  The
//! critical constraint is **zero-copy**: `pread(2)` writes directly into the
//! aligned region, and the same region is immediately GPU-accessible as a Metal
//! `StorageModeShared` buffer — no intermediate bounce buffer required.
//!
//! # Alignment requirement
//!
//! macOS requires that memory passed to
//! `newBufferWithBytesNoCopy:length:options:deallocator:` be aligned to the
//! system's virtual-memory page size.  On Apple Silicon the page size is 16 KB,
//! but large-page mappings use 2 MB boundaries, which is also the natural
//! alignment for MoE expert weights (typically 256 KiB – 4 MiB per expert).
//! Using 2 MB alignment:
//!
//! - Satisfies Metal's page-alignment requirement on all Apple Silicon variants.
//! - Keeps each buffer on a single huge-page, reducing TLB pressure during GPU
//!   scatter-gather DMA.
//! - Allows `pread` to write at full DMA bandwidth without cross-page tearing.
//!
//! # Pool design — double buffering
//!
//! The pool holds `2 * K` buffers (two "slots" per active expert stream):
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Available queue (parking_lot::Mutex<VecDeque<AlignedBuffer>>)   │
//! │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                 │
//! │  │ buf[0] │  │ buf[1] │  │ buf[2] │  │ buf[3] │  …  2*K total  │
//! │  └────────┘  └────────┘  └────────┘  └────────┘                 │
//! └──────────────────────────────────────────────────────────────────┘
//!       │  acquire()                              │  release()
//!       ▼                                         │
//!    pread() ──► GPU kernel ──► done ─────────────┘
//! ```
//!
//! While the GPU processes expert *N* (buffer A), the I/O thread prefetches
//! expert *N+1* into buffer B, so neither side ever stalls.
//!
//! # Thread safety
//!
//! [`ExpertBufferPool`] is `Send + Sync`.  [`AlignedBuffer`] is `Send` but not
//! `Clone`; the pool transfers exclusive ownership on `acquire` and reclaims it
//! on `release`.
//!
//! # Example
//!
//! ```ignore
//! use std::os::unix::io::AsRawFd;
//! use pmetal_metal::{MetalContext, expert_buffer::{ExpertBufferPool, ExpertBufferPoolConfig}};
//!
//! let ctx = MetalContext::global()?;
//! let pool = ExpertBufferPool::new(
//!     &ctx,
//!     ExpertBufferPoolConfig {
//!         buffer_size: 4 * 1024 * 1024, // 4 MiB per expert
//!         k: 4,                          // 4 concurrent expert streams → 8 buffers
//!     },
//! )?;
//!
//! // I/O thread: acquire a buffer, fill it with pread, hand it to the GPU.
//! let mut buf = pool.acquire_blocking();
//! let n = buf.pread(file.as_raw_fd(), file_offset)?;
//! assert_eq!(n, buf.size());
//! // Pass buf.metal_buffer() to your compute encoder …
//!
//! // When done, return the buffer to the pool.
//! pool.release(buf);
//! ```

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use parking_lot::{Condvar, Mutex};
use tracing::{debug, trace};

use crate::context::MetalContext;
use crate::error::{MetalError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// POSIX FFI (inline — avoids adding a `libc` dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// 2 MiB alignment — satisfies Metal's page-alignment requirement and keeps
/// each buffer on a single huge-page boundary.
pub const ALIGN_2MB: usize = 2 * 1024 * 1024;

mod sys {
    use std::ffi::c_void;

    // posix_memalign(3): allocates `size` bytes with `alignment`-byte alignment.
    // Returns 0 on success, an errno value on failure.
    // `alignment` must be a power of two and a multiple of sizeof(void*).
    //
    // pread(2): read `count` bytes from fd at `offset` into `buf` without
    // advancing the file position.  Returns bytes read, or -1 on error.
    unsafe extern "C" {
        pub fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        pub fn free(ptr: *mut c_void);
        pub fn pread(fd: i32, buf: *mut c_void, count: usize, offset: i64) -> isize;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AlignedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A 2 MB-aligned CPU allocation wrapped as a GPU-visible Metal buffer.
///
/// The allocation is made with `posix_memalign(ALIGN_2MB, size)` and registered
/// with Metal via `newBufferWithBytesNoCopy:length:options:deallocator:` using
/// `StorageModeShared`.  No bytes are ever copied: the CPU and GPU share the
/// same physical pages.
///
/// # Ownership
///
/// `AlignedBuffer` owns the underlying allocation.  When dropped:
///
/// 1. The [`Retained`] Metal buffer handle is released (ARC decrement).  Metal
///    will stop accessing the pages once all GPU work referencing them has
///    completed, but the caller is responsible for ensuring the GPU has finished
///    before dropping.
/// 2. The raw allocation is freed with `free(3)`.
///
/// To avoid use-after-free, always wait for GPU completion before dropping or
/// returning the buffer to the pool.
pub struct AlignedBuffer {
    /// Raw 2 MB-aligned allocation.  Kept alive for the lifetime of
    /// `metal_buf` and freed in `Drop`.
    raw: NonNull<std::ffi::c_void>,

    /// Byte capacity of the allocation.
    size: usize,

    /// Metal buffer wrapping `raw` (no-copy, shared storage).
    ///
    /// Wrapped in `ManuallyDrop` so that our `Drop` implementation controls the
    /// exact drop order: Metal buffer ARC is decremented first, then the backing
    /// allocation is freed.  Without `ManuallyDrop`, Rust would auto-drop
    /// `metal_buf` *after* our `Drop::drop` body runs, which means `free` would
    /// execute while Metal still holds a CPU-side retain on the allocation.
    metal_buf: std::mem::ManuallyDrop<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl AlignedBuffer {
    /// Allocate a new 2 MB-aligned buffer and register it with `device` as a
    /// shared Metal buffer.
    ///
    /// # Arguments
    ///
    /// * `device` — the Metal device to register the buffer with.
    /// * `size`   — number of bytes to allocate.  Must be `> 0`.  The actual
    ///   allocation is rounded up to the next multiple of [`ALIGN_2MB`] so that
    ///   the buffer always covers at least `size` bytes and the registration
    ///   satisfies Metal's page-alignment length requirement.
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::BufferCreation`] if:
    ///
    /// - `size` is zero.
    /// - `posix_memalign` fails (ENOMEM / EINVAL).
    /// - `newBufferWithBytesNoCopy` returns `nil` (device out of resources).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(MetalError::BufferCreation {
                size: 0,
                reason: "AlignedBuffer size must be > 0".to_string(),
            });
        }

        // Round up to the next multiple of ALIGN_2MB so that both the
        // allocation size and the Metal buffer length are page-aligned.
        let alloc_size = size.div_ceil(ALIGN_2MB) * ALIGN_2MB;

        // ── allocate ──────────────────────────────────────────────────────────
        let raw = {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

            // SAFETY:
            // 1. `ptr` is a valid stack variable whose address we pass.
            // 2. `ALIGN_2MB` is a power of two and a multiple of `sizeof(void*)`.
            // 3. `alloc_size` is a non-zero multiple of `ALIGN_2MB`.
            let rc = unsafe { sys::posix_memalign(&mut ptr, ALIGN_2MB, alloc_size) };
            if rc != 0 {
                return Err(MetalError::BufferCreation {
                    size: alloc_size,
                    reason: format!(
                        "posix_memalign({ALIGN_2MB}, {alloc_size}) failed with errno {rc}"
                    ),
                });
            }

            // SAFETY: `posix_memalign` returned 0, so `ptr` is now a valid
            // non-null pointer to `alloc_size` bytes of writable memory.
            NonNull::new(ptr).ok_or_else(|| MetalError::BufferCreation {
                size: alloc_size,
                reason: "posix_memalign returned null despite rc == 0".to_string(),
            })?
        };

        // ── register with Metal (no-copy) ────────────────────────────────────
        //
        // `StorageModeShared`: unified memory — both CPU and GPU can read/write
        // the same physical pages.
        //
        // `HazardTrackingModeTracked`: Metal automatically inserts barriers when
        // buffers transition between CPU and GPU access, preventing data races
        // without manual fencing in the common case.
        //
        // `deallocator: None`: we own the allocation and free it in `Drop`.
        // We must not free it before all GPU work referencing this buffer has
        // completed; the pool's `release` / `acquire_blocking` protocol enforces
        // this by requiring callers to wait for GPU completion first.
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked;

        // SAFETY:
        // 1. `raw.as_ptr()` is non-null, 2 MB-aligned, and valid for
        //    `alloc_size` bytes — guaranteed by `posix_memalign` above.
        // 2. `alloc_size` matches the allocation length exactly.
        // 3. `options` requests shared storage, which is valid for unified
        //    memory on Apple Silicon.
        // 4. `deallocator` is `None`: Metal will not attempt to free the memory;
        //    our `Drop` implementation calls `free` after the Metal buffer handle
        //    (and thus all GPU references) have been released.
        let metal_buf = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new_unchecked(raw.as_ptr()),
                alloc_size,
                options,
                None, // deallocator: we handle lifetime in Drop
            )
        }
        .ok_or_else(|| MetalError::BufferCreation {
            size: alloc_size,
            reason: "newBufferWithBytesNoCopy returned nil — device out of resources?".to_string(),
        })?;

        debug!(
            size_requested = size,
            alloc_size,
            ptr = ?raw.as_ptr(),
            "AlignedBuffer allocated"
        );

        Ok(Self {
            raw,
            size: alloc_size,
            metal_buf: std::mem::ManuallyDrop::new(metal_buf),
        })
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    /// The total byte capacity of the allocation (rounded up from the requested
    /// size to the next multiple of [`ALIGN_2MB`]).
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// A reference to the underlying Metal buffer.
    ///
    /// Pass this to `MTLComputeCommandEncoder::setBuffer` or equivalent when
    /// dispatching a GPU kernel over the expert weights.
    #[inline]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.metal_buf
    }

    /// A retained (ref-counted) clone of the Metal buffer handle.
    ///
    /// Use this when the kernel dispatch outlives the `AlignedBuffer` borrow
    /// (e.g., when passing into a closure or async task).
    #[inline]
    pub fn metal_buffer_retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        (*self.metal_buf).clone()
    }

    /// A raw byte pointer to the beginning of the aligned allocation.
    ///
    /// Suitable for passing to `pread(2)` directly.  The returned pointer is
    /// valid for [`Self::size`] bytes of reads and writes.
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent GPU access is happening to this
    /// region, and that the data written is valid for the intended interpretation
    /// before submitting GPU work.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.raw.as_ptr() as *mut u8
    }

    /// An immutable byte slice covering the full allocation.
    ///
    /// Safe for read-only inspection once the caller has synchronized any
    /// in-flight GPU writes that might target this buffer.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `raw` is valid for `size` bytes for the lifetime of `self`,
        // and we only expose immutable access here.
        unsafe { std::slice::from_raw_parts(self.raw.as_ptr() as *const u8, self.size) }
    }

    /// A mutable byte slice covering the full allocation.
    ///
    /// # Safety
    ///
    /// Same as [`Self::as_ptr`]: the GPU must not be concurrently reading or
    /// writing this buffer.
    #[inline]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `raw` is valid for `size` bytes and exclusively owned by
        // this struct at the point where the caller holds `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.raw.as_ptr() as *mut u8, self.size) }
    }

    /// Copy a byte slice into the start of the buffer.
    ///
    /// This is a convenience wrapper for prefetched-expert paths that already
    /// have the expert bytes resident in CPU memory and want to preserve the
    /// aligned Metal-shared fast path for subsequent GPU dispatch.
    pub fn write_prefix(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.size {
            return Err(MetalError::InvalidConfig(format!(
                "write_prefix: source length {} exceeds buffer size {}",
                src.len(),
                self.size
            )));
        }

        // SAFETY: `self` is exclusively borrowed and no concurrent GPU access
        // is permitted while the caller holds `&mut self`.
        let dst = unsafe { self.as_bytes_mut() };
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Return a mutable view into the first `len` bytes of the buffer.
    ///
    /// This is the safe counterpart to [`Self::as_bytes_mut`] for callers that
    /// only need to fill a bounded prefix before handing the buffer to Metal.
    pub fn prefix_mut(&mut self, len: usize) -> Result<&mut [u8]> {
        if len > self.size {
            return Err(MetalError::InvalidConfig(format!(
                "prefix_mut: requested prefix {} exceeds buffer size {}",
                len, self.size
            )));
        }

        // SAFETY: `self` is exclusively borrowed and the returned slice is
        // bounded to `len <= self.size`.
        let bytes = unsafe { self.as_bytes_mut() };
        Ok(&mut bytes[..len])
    }

    // ── I/O ───────────────────────────────────────────────────────────────────

    /// Fill the buffer from a file using `pread(2)` at the given byte offset.
    ///
    /// Reads exactly [`Self::size`] bytes from `fd` starting at `file_offset`
    /// directly into the aligned memory.  Because the memory is Metal-shared,
    /// there is no copy step; the GPU can access the data immediately after this
    /// call returns.
    ///
    /// # Arguments
    ///
    /// * `fd`          — file descriptor open for reading.
    /// * `file_offset` — byte offset in the file at which to start reading.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.  This equals [`Self::size`] unless the
    /// file is shorter than expected, in which case the trailing bytes of the
    /// buffer are left unchanged (typically zeroed from initialization).
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::ExecutionFailed`] if the underlying `pread` syscall
    /// returns `-1`.
    pub fn pread(&mut self, fd: i32, file_offset: u64) -> Result<usize> {
        let count = self.size;
        let dst = self.raw.as_ptr();

        // SAFETY:
        // 1. `dst` is valid for `count` bytes of writes (guaranteed by
        //    `posix_memalign` with `alloc_size == self.size`).
        // 2. `fd` is a caller-supplied file descriptor; we trust the caller to
        //    provide a valid, readable fd.
        // 3. `file_offset` is cast to `i64`; offsets ≥ 2^63 are unsupported but
        //    unrealistic for expert weight files.
        // 4. `pread` does not advance the file position, so concurrent reads on
        //    the same fd from other threads are safe (the fd itself is not
        //    mutated by pread).
        let n = unsafe { sys::pread(fd, dst, count, file_offset as i64) };

        if n < 0 {
            // `pread` returns -1 and sets errno on error.  We capture the errno
            // value via `std::io::Error::last_os_error()` for a human-readable
            // message without adding extra dependencies.
            let err = std::io::Error::last_os_error();
            return Err(MetalError::ExecutionFailed(format!(
                "pread(fd={fd}, offset={file_offset}, count={count}) failed: {err}"
            )));
        }

        trace!(
            fd,
            file_offset,
            requested = count,
            read = n,
            "AlignedBuffer::pread"
        );

        Ok(n as usize)
    }

    /// Fill a sub-range of the buffer from a file using `pread(2)`.
    ///
    /// Reads up to `byte_len` bytes from `fd` at `file_offset` into the buffer
    /// starting at `buf_offset`.
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::InvalidConfig`] if the range `[buf_offset,
    /// buf_offset + byte_len)` extends beyond [`Self::size`], or
    /// [`MetalError::ExecutionFailed`] on `pread` failure.
    pub fn pread_range(
        &mut self,
        fd: i32,
        file_offset: u64,
        buf_offset: usize,
        byte_len: usize,
    ) -> Result<usize> {
        let end = buf_offset.checked_add(byte_len).ok_or_else(|| {
            MetalError::InvalidConfig("buf_offset + byte_len overflows usize".to_string())
        })?;
        if end > self.size {
            return Err(MetalError::InvalidConfig(format!(
                "pread_range: range [{}..{}) exceeds buffer size {}",
                buf_offset, end, self.size
            )));
        }

        // SAFETY: `dst` is within the valid allocation as checked above.
        let dst =
            unsafe { (self.raw.as_ptr() as *mut u8).add(buf_offset) } as *mut std::ffi::c_void;

        let n = unsafe { sys::pread(fd, dst, byte_len, file_offset as i64) };

        if n < 0 {
            let err = std::io::Error::last_os_error();
            return Err(MetalError::ExecutionFailed(format!(
                "pread(fd={fd}, offset={file_offset}, count={byte_len}) failed: {err}"
            )));
        }

        Ok(n as usize)
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let raw_ptr = self.raw.as_ptr();

        // Step 1: Release the Metal buffer CPU-side ARC retain.
        //
        // `self.metal_buf` is a `ManuallyDrop<Retained<…>>`, so Rust will never
        // auto-drop it — we must do so explicitly here.  This decrements the
        // Objective-C retain count.  After this point, the Metal object is kept
        // alive only by any in-flight command buffers that themselves hold a
        // retain (via the GPU driver's internal bookkeeping).  Once those
        // command buffers complete, Metal releases its retain and the MTLBuffer
        // object is deallocated — but it will no longer attempt DMA into the
        // backing allocation because Metal's hazard tracking will have already
        // flushed all pending work before the command buffer completes.
        //
        // The caller is contractually responsible for waiting on GPU completion
        // before dropping `AlignedBuffer`.
        //
        // SAFETY: `self.metal_buf` was initialized in `AlignedBuffer::new` and
        // has not been dropped before (it is `ManuallyDrop`).  Calling
        // `ManuallyDrop::drop` exactly once here is correct.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.metal_buf) };

        // Step 2: Free the backing allocation.
        //
        // SAFETY:
        // 1. `raw_ptr` was returned by `posix_memalign`, so `free(3)` is the
        //    correct deallocation call.
        // 2. We have released the Metal CPU-side retain above; the allocation is
        //    no longer reachable from Metal on the CPU side.
        // 3. `raw_ptr` has not been freed before (we own the allocation
        //    exclusively for its entire lifetime).
        unsafe { sys::free(raw_ptr) };

        debug!(ptr = ?raw_ptr, "AlignedBuffer freed");
    }
}

// SAFETY: `AlignedBuffer` owns a raw allocation and an `Retained<MTLBuffer>`.
// `Retained` is `Send` in the objc2 ecosystem (MTLBuffer is thread-safe at the
// Objective-C ARC level).  We never hand out raw mutable references to the
// allocation without requiring exclusive access (`&mut self` or ownership).
unsafe impl Send for AlignedBuffer {}

// `AlignedBuffer` is NOT `Sync` by default because its `as_ptr()` / `pread`
// methods allow mutable access to the backing allocation — allowing shared
// references to mutate it would violate Rust's aliasing rules.  Users who need
// to share a buffer across threads must use their own synchronization (e.g.,
// wrapping in `Mutex`).  The pool itself is `Sync` by using a `Mutex` internally.

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("ptr", &self.raw.as_ptr())
            .field("size", &self.size)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertBufferPool
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ExpertBufferPool`].
#[derive(Debug, Clone)]
pub struct ExpertBufferPoolConfig {
    /// Byte capacity of each individual buffer.
    ///
    /// Typically the size of one expert's weight tensor, rounded up so that
    /// every expert fits in a single buffer.  The actual allocation will be
    /// rounded up to the next multiple of [`ALIGN_2MB`].
    pub buffer_size: usize,

    /// Number of *concurrent expert streams* (`K`).
    ///
    /// The pool allocates `2 * K` buffers so that one buffer per stream can be
    /// in-flight on the GPU while the other is being filled by `pread`.
    ///
    /// A value of 1 gives simple ping-pong double buffering; higher values
    /// support deeper prefetch pipelines.
    pub k: usize,
}

impl ExpertBufferPoolConfig {
    /// Total number of buffers in the pool (`2 * K`).
    #[inline]
    pub fn total_buffers(&self) -> usize {
        2 * self.k
    }
}

/// Pre-allocated pool of 2 MB-aligned, GPU-visible buffers for expert weight
/// offloading.
///
/// See the [module documentation](self) for an overview of the design.
///
/// # Acquiring and releasing buffers
///
/// - [`acquire_blocking`](ExpertBufferPool::acquire_blocking) — block until a
///   buffer is available.
/// - [`try_acquire`](ExpertBufferPool::try_acquire) — non-blocking attempt;
///   returns `None` if the pool is empty.
/// - [`release`](ExpertBufferPool::release) — return a buffer to the pool after
///   GPU work has completed.
///
/// # Memory accounting
///
/// The pool eagerly allocates all `2 * K` buffers at construction time so that
/// the first `acquire` call never triggers a large allocation on the critical
/// path.
pub struct ExpertBufferPool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    /// Mutex-protected queue of available buffers.
    ///
    /// [`Condvar`] is used so that `acquire_blocking` can sleep without
    /// spinning when the pool is temporarily exhausted.
    available: Mutex<VecDeque<AlignedBuffer>>,

    /// Condition variable signalled when a buffer is returned to the pool.
    returned: Condvar,

    /// Total number of buffers managed by this pool.
    total: usize,

    /// Byte capacity of each individual buffer (after alignment rounding).
    buffer_size: usize,
}

impl ExpertBufferPool {
    /// Create a new pool and eagerly allocate all `2 * K` buffers.
    ///
    /// # Arguments
    ///
    /// * `ctx`    — Metal context whose device the buffers are registered with.
    /// * `config` — Pool configuration (buffer size and concurrency factor `K`).
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer allocation or Metal registration fails.
    /// Partial allocations are cleaned up automatically when the partially-
    /// constructed pool is dropped.
    pub fn new(ctx: &MetalContext, config: ExpertBufferPoolConfig) -> Result<Self> {
        if config.buffer_size == 0 {
            return Err(MetalError::InvalidConfig(
                "ExpertBufferPoolConfig::buffer_size must be > 0".to_string(),
            ));
        }
        if config.k == 0 {
            return Err(MetalError::InvalidConfig(
                "ExpertBufferPoolConfig::k must be > 0 (double-buffering requires at least 1 stream)".to_string(),
            ));
        }

        let total = config.total_buffers();
        let device = ctx.device();

        let mut queue: VecDeque<AlignedBuffer> = VecDeque::with_capacity(total);
        for i in 0..total {
            let buf = AlignedBuffer::new(device, config.buffer_size).map_err(|e| {
                MetalError::BufferCreation {
                    size: config.buffer_size,
                    reason: format!("Failed to allocate expert buffer {i}/{total}: {e}"),
                }
            })?;
            queue.push_back(buf);
        }

        // Capture the actual allocation size (may be larger than requested due
        // to ALIGN_2MB rounding) from the first buffer.
        let actual_size = queue[0].size();

        debug!(
            total,
            buffer_size = actual_size,
            total_bytes = total * actual_size,
            "ExpertBufferPool initialized"
        );

        Ok(Self {
            inner: Arc::new(PoolInner {
                available: Mutex::new(queue),
                returned: Condvar::new(),
                total,
                buffer_size: actual_size,
            }),
        })
    }

    /// Block until a buffer is available, then return it.
    ///
    /// If all `2 * K` buffers are currently checked out, this call sleeps on a
    /// condition variable until one is returned via [`release`](Self::release).
    ///
    /// # Deadlock warning
    ///
    /// Do not hold any lock that is also taken inside [`release`] while calling
    /// this method.
    pub fn acquire_blocking(&self) -> AlignedBuffer {
        let mut guard = self.inner.available.lock();
        loop {
            if let Some(buf) = guard.pop_front() {
                trace!(remaining = guard.len(), "ExpertBufferPool: buffer acquired");
                return buf;
            }
            // Queue is empty — wait for a release() notification.
            self.inner.returned.wait(&mut guard);
        }
    }

    /// Attempt to acquire a buffer without blocking.
    ///
    /// Returns `Some(buffer)` if one is immediately available, or `None` if all
    /// buffers are currently checked out.
    pub fn try_acquire(&self) -> Option<AlignedBuffer> {
        let mut guard = self.inner.available.lock();
        let buf = guard.pop_front();
        if buf.is_some() {
            trace!(
                remaining = guard.len(),
                "ExpertBufferPool: buffer acquired (try)"
            );
        }
        buf
    }

    /// Return a buffer to the pool.
    ///
    /// # Contract
    ///
    /// The caller **must** ensure that all GPU work using this buffer has
    /// completed before calling `release`.  The simplest way is to call
    /// `commandBuffer.waitUntilCompleted()` before releasing.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `buf.size()` does not match the pool's
    /// configured `buffer_size`, which would indicate that a buffer from a
    /// different pool or allocation is being returned.
    pub fn release(&self, buf: AlignedBuffer) {
        debug_assert_eq!(
            buf.size(),
            self.inner.buffer_size,
            "ExpertBufferPool::release: buffer size {} does not match pool size {}",
            buf.size(),
            self.inner.buffer_size,
        );

        {
            let mut guard = self.inner.available.lock();
            guard.push_back(buf);
            trace!(available = guard.len(), "ExpertBufferPool: buffer released");
        }
        // Wake one waiter (if any).
        self.inner.returned.notify_one();
    }

    /// Total number of buffers managed by this pool (`2 * K`).
    #[inline]
    pub fn total_buffers(&self) -> usize {
        self.inner.total
    }

    /// Byte capacity of each individual buffer (after 2 MB alignment rounding).
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    /// Total memory held by this pool in bytes (`total_buffers * buffer_size`).
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.inner.total * self.inner.buffer_size
    }

    /// Number of buffers currently available for acquisition (snapshot).
    ///
    /// This is a best-effort, non-synchronized snapshot suitable for diagnostics
    /// and logging; do not use it to make control-flow decisions without
    /// additional synchronization.
    pub fn available_count(&self) -> usize {
        self.inner.available.lock().len()
    }

    /// Number of buffers currently checked out (snapshot).
    pub fn in_flight_count(&self) -> usize {
        self.inner.total - self.available_count()
    }
}

// SAFETY: `ExpertBufferPool` wraps an `Arc<PoolInner>` which itself contains
// a `Mutex`-protected `VecDeque` of `AlignedBuffer`s.  `AlignedBuffer: Send`,
// `Mutex<VecDeque<AlignedBuffer>>: Send + Sync`, and `Arc<T>: Send + Sync`
// when `T: Send + Sync`.  Therefore `ExpertBufferPool` is `Send + Sync`.
unsafe impl Send for ExpertBufferPool {}
unsafe impl Sync for ExpertBufferPool {}

impl Clone for ExpertBufferPool {
    /// Clone returns a new handle to the *same* pool (shared ownership via
    /// `Arc`).  Buffers are shared across all clones.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for ExpertBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertBufferPool")
            .field("total_buffers", &self.inner.total)
            .field("buffer_size", &self.inner.buffer_size)
            .field("available", &self.available_count())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::MetalContext;

    fn make_pool(k: usize, buf_size: usize) -> ExpertBufferPool {
        let ctx = MetalContext::new().expect("Metal context");
        ExpertBufferPool::new(
            &ctx,
            ExpertBufferPoolConfig {
                buffer_size: buf_size,
                k,
            },
        )
        .expect("pool creation")
    }

    // ── AlignedBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_aligned_buffer_allocation() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 4 * 1024 * 1024).unwrap();

        // Allocation is rounded up to 2 MB boundary.
        assert_eq!(buf.size() % ALIGN_2MB, 0);
        assert!(buf.size() >= 4 * 1024 * 1024);

        // The raw pointer must satisfy our alignment contract.
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % ALIGN_2MB, 0, "pointer is not 2 MB-aligned");
    }

    #[test]
    fn test_aligned_buffer_size_rounding() {
        let ctx = MetalContext::new().unwrap();

        // A 1-byte request should be rounded up to exactly ALIGN_2MB.
        let buf = AlignedBuffer::new(ctx.device(), 1).unwrap();
        assert_eq!(buf.size(), ALIGN_2MB);

        // An exact multiple should stay the same.
        let buf2 = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();
        assert_eq!(buf2.size(), ALIGN_2MB);

        // ALIGN_2MB + 1 should round up to 2 * ALIGN_2MB.
        let buf3 = AlignedBuffer::new(ctx.device(), ALIGN_2MB + 1).unwrap();
        assert_eq!(buf3.size(), 2 * ALIGN_2MB);
    }

    #[test]
    fn test_aligned_buffer_zero_rejected() {
        let ctx = MetalContext::new().unwrap();
        let result = AlignedBuffer::new(ctx.device(), 0);
        assert!(result.is_err(), "zero-size allocation should fail");
    }

    #[test]
    fn test_aligned_buffer_metal_buffer() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();
        // Verify the Metal buffer has the expected length.
        assert_eq!(buf.metal_buffer().length(), buf.size());
    }

    #[test]
    fn test_aligned_buffer_write_read() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();

        // Write a pattern into the aligned memory via the raw pointer.
        // SAFETY: no concurrent GPU access in this test.
        let slice = unsafe { buf.as_bytes_mut() };
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }

        // Read back and verify.
        let slice2 = unsafe { buf.as_bytes_mut() };
        for (i, &byte) in slice2.iter().enumerate() {
            assert_eq!(byte, (i % 251) as u8);
        }
    }

    #[test]
    fn test_aligned_buffer_write_prefix() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();
        let payload = [9u8, 7, 5, 3, 1];
        buf.write_prefix(&payload).unwrap();

        // SAFETY: no concurrent GPU access in this test.
        let slice = unsafe { buf.as_bytes_mut() };
        assert_eq!(&slice[..payload.len()], &payload);
    }

    #[test]
    fn test_pread_from_file() {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");

        // Write 4 MB of known data.
        let data: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 199) as u8).collect();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 4 * 1024 * 1024).unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = tmp.as_raw_fd();
        let n = buf.pread(fd, 0).unwrap();
        assert_eq!(n, buf.size()); // entire buffer should be filled

        // Verify the first 4 MB of content.
        let slice = unsafe { buf.as_bytes_mut() };
        for (i, &byte) in slice[..data.len()].iter().enumerate() {
            assert_eq!(
                byte, data[i],
                "mismatch at byte {i}: got {byte}, want {}",
                data[i]
            );
        }
    }

    #[test]
    fn test_pread_range() {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let payload = b"hello expert weights";
        tmp.write_all(payload).unwrap();
        tmp.flush().unwrap();

        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();

        // Zero the buffer first.
        unsafe { buf.as_bytes_mut() }.fill(0u8);

        use std::os::unix::io::AsRawFd;
        let n = buf
            .pread_range(tmp.as_raw_fd(), 0, 0, payload.len())
            .unwrap();
        assert_eq!(n, payload.len());

        let slice = unsafe { buf.as_bytes_mut() };
        assert_eq!(&slice[..payload.len()], payload);
    }

    #[test]
    fn test_pread_range_out_of_bounds() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();
        // Attempt to read past the end of the buffer.
        let result = buf.pread_range(0, 0, buf.size() - 1, 2);
        assert!(result.is_err(), "out-of-bounds range should be rejected");
    }

    // ── ExpertBufferPool ──────────────────────────────────────────────────────

    #[test]
    fn test_pool_creation() {
        let pool = make_pool(4, 4 * 1024 * 1024);
        assert_eq!(pool.total_buffers(), 8);
        assert_eq!(pool.available_count(), 8);
        assert_eq!(pool.in_flight_count(), 0);
    }

    #[test]
    fn test_pool_invalid_config() {
        let ctx = MetalContext::new().unwrap();

        let r1 = ExpertBufferPool::new(
            &ctx,
            ExpertBufferPoolConfig {
                buffer_size: 0,
                k: 2,
            },
        );
        assert!(r1.is_err(), "zero buffer_size should be rejected");

        let r2 = ExpertBufferPool::new(
            &ctx,
            ExpertBufferPoolConfig {
                buffer_size: ALIGN_2MB,
                k: 0,
            },
        );
        assert!(r2.is_err(), "k=0 should be rejected");
    }

    #[test]
    fn test_pool_acquire_release_cycle() {
        let pool = make_pool(2, ALIGN_2MB);
        assert_eq!(pool.available_count(), 4);

        let b0 = pool.acquire_blocking();
        assert_eq!(pool.available_count(), 3);
        assert_eq!(pool.in_flight_count(), 1);

        let b1 = pool.acquire_blocking();
        assert_eq!(pool.available_count(), 2);

        pool.release(b0);
        assert_eq!(pool.available_count(), 3);

        pool.release(b1);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_pool_try_acquire_exhaustion() {
        let pool = make_pool(1, ALIGN_2MB); // 2 total buffers
        let b0 = pool.try_acquire().expect("first acquire");
        let b1 = pool.try_acquire().expect("second acquire");
        let b2 = pool.try_acquire();
        assert!(b2.is_none(), "pool should be exhausted after 2 acquires");

        pool.release(b0);
        pool.release(b1);

        let b3 = pool.try_acquire();
        assert!(b3.is_some(), "should be able to acquire after release");
        pool.release(b3.unwrap());
    }

    #[test]
    fn test_pool_clone_shares_state() {
        let pool = make_pool(1, ALIGN_2MB);
        let pool2 = pool.clone();

        let buf = pool.acquire_blocking();
        assert_eq!(pool2.available_count(), 1); // clone sees the change

        pool2.release(buf);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_pool_blocking_acquire_from_thread() {
        use std::sync::Arc as StdArc;
        use std::time::Duration;

        let pool = StdArc::new(make_pool(1, ALIGN_2MB));
        // Drain the pool.
        let b0 = pool.acquire_blocking();
        let b1 = pool.acquire_blocking();

        let pool2 = StdArc::clone(&pool);
        let handle = std::thread::spawn(move || {
            // This will block until we release a buffer below.
            let buf = pool2.acquire_blocking();
            pool2.release(buf);
        });

        // Give the thread time to start and block.
        std::thread::sleep(Duration::from_millis(10));

        // Unblock the waiter.
        pool.release(b0);
        pool.release(b1);

        handle.join().expect("thread panicked");
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_pool_memory_accounting() {
        let pool = make_pool(3, 4 * 1024 * 1024);
        // 6 buffers, each rounded up to ALIGN_2MB (4 MiB ≤ ALIGN_2MB, so stays 4 MiB)
        // Actually 4 MiB < 2 MiB is false; 4 MiB = 2 * 2 MiB = 2 * ALIGN_2MB? No:
        // ALIGN_2MB = 2 MiB = 2*1024*1024; 4 MiB = 4*1024*1024 = 2 * ALIGN_2MB.
        let expected_buf_size = 2 * ALIGN_2MB; // 4 MiB rounds up to 4 MiB (exact multiple)
        assert_eq!(pool.buffer_size(), expected_buf_size);
        assert_eq!(pool.total_bytes(), 6 * expected_buf_size);
    }

    #[test]
    fn test_pool_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExpertBufferPool>();
    }

    #[test]
    fn test_aligned_buffer_debug_format() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), ALIGN_2MB).unwrap();
        let s = format!("{buf:?}");
        assert!(s.contains("AlignedBuffer"), "debug format: {s}");
    }

    #[test]
    fn test_pool_debug_format() {
        let pool = make_pool(2, ALIGN_2MB);
        let s = format!("{pool:?}");
        assert!(s.contains("ExpertBufferPool"), "debug format: {s}");
    }
}
