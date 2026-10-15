#![allow(unsafe_code)]

//! Page-aligned buffer pool for expert weight offloading.
//!
//! Expert weight offloading streams quantized MoE expert weights from disk into
//! GPU-visible memory at decode time.  The critical constraint is
//! **zero-copy on the GPU side**: `pread(2)` writes directly into a
//! page-aligned region, and the same region is immediately GPU-accessible as a
//! Metal `StorageModeShared` buffer — no bounce buffer between the read and the
//! kernel.
//!
//! # What the pool is, and is not
//!
//! The pool is a bounded set of destination buffers, not a cache.  It keeps no
//! expert resident across uses and evicts nothing: the operating system's page
//! cache already holds recently read expert files in free memory and serves a
//! repeat read at memory speed, which no cache of our own can beat while
//! sharing that memory with it.  The buffers are read into with plain
//! `pread` (never `F_NOCACHE`) so that the page cache keeps working.
//!
//! Reading from the SSD and running GPU kernels do not overlap profitably on
//! Apple Silicon: both draw on the same unified memory and the same fabric.
//! The second buffer of each pair exists so the *next* expert's read can be
//! issued while the current kernel is encoded, not because the transfer hides
//! behind compute.
//!
//! # Alignment requirement
//!
//! `newBufferWithBytesNoCopy:length:options:deallocator:` needs memory aligned
//! to the system page size, with a length that is a multiple of it (16 KiB on
//! Apple Silicon).  Buffers are anonymous page mappings: page-aligned,
//! zero-filled, and committed lazily, so a buffer larger than the expert it
//! carries costs only the pages that are touched.
//!
//! # Pool design
//!
//! The pool holds `2 * K` buffers:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Available queue (parking_lot::Mutex<VecDeque<AlignedBuffer>>)   │
//! │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                 │
//! │  │ buf[0] │  │ buf[1] │  │ buf[2] │  │ buf[3] │  …  2*K total  │
//! │  └────────┘  └────────┘  └────────┘  └────────┘                 │
//! └──────────────────────────────────────────────────────────────────┘
//!       │  acquire()                              │  release() / drop
//!       ▼                                         │
//!    pread() ──► GPU kernel ──► done ─────────────┘
//! ```
//!
//! A buffer taken from the pool returns to it when it is released **or
//! dropped**, so an error path or a panic between `acquire` and `release`
//! cannot shrink the pool until `acquire_blocking` waits forever.  Either way
//! the caller must have waited for the GPU to finish with it first.
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
//! // Pass buf.metal_buffer() to your compute encoder …
//!
//! // When done, return the buffer to the pool.
//! pool.release(buf);
//! ```

use std::collections::VecDeque;
use std::mem::ManuallyDrop;
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

mod sys {
    use std::ffi::c_void;

    pub const PROT_READ: i32 = 0x1;
    pub const PROT_WRITE: i32 = 0x2;
    pub const MAP_PRIVATE: i32 = 0x0002;
    pub const MAP_ANON: i32 = 0x1000;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    pub const EINTR: i32 = 4;

    // mmap(2) with MAP_ANON: page-aligned, zero-filled, lazily committed.
    // pread(2): read `count` bytes from fd at `offset` into `buf` without
    // advancing the file position.  Returns bytes read, or -1 on error.
    unsafe extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn pread(fd: i32, buf: *mut c_void, count: usize, offset: i64) -> isize;
        pub fn getpagesize() -> i32;
    }
}

/// The system page size in bytes (16 KiB on Apple Silicon).
pub fn page_size() -> usize {
    // SAFETY: `getpagesize` takes no arguments and has no preconditions.
    let size = unsafe { sys::getpagesize() };
    usize::try_from(size)
        .ok()
        .filter(|s| s.is_power_of_two())
        .unwrap_or(16 * 1024)
}

/// The most bytes one `pread` call is asked for; macOS rejects a count above
/// `INT_MAX`.
const MAX_READ_CHUNK: usize = 1 << 30;

/// Read until `len` bytes arrive or the file ends, retrying calls a signal
/// interrupted and continuing after short reads.
///
/// `read_at(dst, count, offset)` is one `pread`.  Returns the bytes read,
/// which is less than `len` only at end of file.
fn read_fully(
    mut read_at: impl FnMut(*mut u8, usize, u64) -> std::io::Result<usize>,
    dst: *mut u8,
    len: usize,
    offset: u64,
) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < len {
        let position = offset.checked_add(done as u64).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "file offset overflows")
        })?;
        // SAFETY: `done < len`, and the caller guarantees `dst` is valid for `len` bytes.
        let at = unsafe { dst.add(done) };
        match read_at(at, (len - done).min(MAX_READ_CHUNK), position) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(error) if error.raw_os_error() == Some(sys::EINTR) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(done)
}

/// `pread(2)` as an `io::Result`.
fn pread_syscall(fd: i32, dst: *mut u8, count: usize, offset: u64) -> std::io::Result<usize> {
    let offset = i64::try_from(offset).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "file offset exceeds i64")
    })?;
    // SAFETY: the caller of `read_fully` guarantees `dst` is valid for
    // `count` writable bytes; `fd` is an integer the kernel validates.
    let n = unsafe { sys::pread(fd, dst.cast(), count, offset) };
    usize::try_from(n).map_err(|_| std::io::Error::last_os_error())
}

// ─────────────────────────────────────────────────────────────────────────────
// Allocation
// ─────────────────────────────────────────────────────────────────────────────

/// An anonymous page mapping wrapped as a no-copy Metal buffer.
///
/// Dropping releases the Metal buffer first, then unmaps the memory it
/// aliased.
struct Allocation {
    /// Start of the mapping.
    raw: NonNull<std::ffi::c_void>,
    /// Length of the mapping in bytes, a multiple of the page size.
    size: usize,
    /// Metal buffer over `raw`.  `ManuallyDrop` so `Drop` can release it
    /// strictly before the memory goes away.
    metal_buf: ManuallyDrop<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Allocation {
    fn new(device: &ProtocolObject<dyn MTLDevice>, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(MetalError::BufferCreation {
                size: 0,
                reason: "AlignedBuffer size must be > 0".to_string(),
            });
        }

        // Round up so the mapping and the Metal buffer length are both whole pages.
        let page = page_size();
        let alloc_size =
            size.checked_next_multiple_of(page)
                .ok_or_else(|| MetalError::BufferCreation {
                    size,
                    reason: "size rounded up to a page overflows".to_string(),
                })?;

        // SAFETY: an anonymous private mapping at a kernel-chosen address
        // (`addr` null, `fd` -1); no existing memory is touched.
        let ptr = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                alloc_size,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_PRIVATE | sys::MAP_ANON,
                -1,
                0,
            )
        };
        let raw = if ptr == sys::MAP_FAILED {
            None
        } else {
            NonNull::new(ptr)
        }
        .ok_or_else(|| MetalError::BufferCreation {
            size: alloc_size,
            reason: format!("mmap failed: {}", std::io::Error::last_os_error()),
        })?;

        // `StorageModeShared`: unified memory, CPU and GPU share the pages.
        // `HazardTrackingModeTracked`: Metal orders GPU access to the buffer.
        let options =
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeTracked;

        // SAFETY: `raw` is page-aligned and valid for `alloc_size` bytes
        // (a whole number of pages), as `newBufferWithBytesNoCopy` requires.
        // `deallocator` is `None` because `Drop` unmaps the memory after
        // releasing the Metal buffer.
        let metal_buf = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(raw, alloc_size, options, None)
        };
        let Some(metal_buf) = metal_buf else {
            // SAFETY: the mapping was made above and nothing else refers to it.
            unsafe { sys::munmap(raw.as_ptr(), alloc_size) };
            return Err(MetalError::BufferCreation {
                size: alloc_size,
                reason: "newBufferWithBytesNoCopy returned nil — device out of resources?"
                    .to_string(),
            });
        };

        debug!(
            size_requested = size,
            alloc_size,
            ptr = ?raw.as_ptr(),
            "AlignedBuffer allocated"
        );

        Ok(Self {
            raw,
            size: alloc_size,
            metal_buf: ManuallyDrop::new(metal_buf),
        })
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: `metal_buf` is initialised in `new` and dropped only here.
        // Releasing it first means the Metal object no longer refers to the
        // pages by the time they are unmapped; the owner is contractually
        // responsible for having waited for any GPU work that used them.
        unsafe { ManuallyDrop::drop(&mut self.metal_buf) };

        // SAFETY: `raw`/`size` describe exactly the mapping `mmap` returned,
        // not yet unmapped.
        unsafe { sys::munmap(self.raw.as_ptr(), self.size) };

        debug!(ptr = ?self.raw.as_ptr(), "AlignedBuffer freed");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AlignedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A page-aligned CPU allocation wrapped as a GPU-visible Metal buffer.
///
/// The memory is an anonymous mapping (zero-filled until written), registered
/// with Metal via `newBufferWithBytesNoCopy:length:options:deallocator:` using
/// `StorageModeShared`.  No bytes are ever copied: the CPU and GPU share the
/// same physical pages.
///
/// # Ownership
///
/// `AlignedBuffer` owns the underlying allocation.  A buffer acquired from an
/// [`ExpertBufferPool`] goes back to that pool when dropped; any other buffer
/// releases the Metal handle and unmaps the memory.  Either way, always wait
/// for GPU completion before dropping or releasing the buffer: the GPU reads
/// these pages directly.
pub struct AlignedBuffer {
    /// The mapping and its Metal buffer.  Taken out on drop.
    allocation: ManuallyDrop<Allocation>,

    /// The pool this buffer returns to when dropped, set while it is checked out.
    home: Option<Arc<PoolInner>>,
}

impl AlignedBuffer {
    /// Allocate a new page-aligned buffer and register it with `device` as a
    /// shared Metal buffer.
    ///
    /// # Arguments
    ///
    /// * `device` — the Metal device to register the buffer with.
    /// * `size`   — number of bytes to allocate.  Must be `> 0`.  The actual
    ///   allocation is rounded up to a whole number of pages
    ///   ([`page_size`]) so that the registration satisfies Metal's
    ///   page-alignment length requirement.
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::BufferCreation`] if:
    ///
    /// - `size` is zero.
    /// - the mapping cannot be made.
    /// - `newBufferWithBytesNoCopy` returns `nil` (device out of resources).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, size: usize) -> Result<Self> {
        Ok(Self {
            allocation: ManuallyDrop::new(Allocation::new(device, size)?),
            home: None,
        })
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    /// The total byte capacity of the allocation (rounded up from the requested
    /// size to a whole number of pages).
    #[inline]
    pub fn size(&self) -> usize {
        self.allocation.size
    }

    /// A reference to the underlying Metal buffer.
    ///
    /// Pass this to `MTLComputeCommandEncoder::setBuffer` or equivalent when
    /// dispatching a GPU kernel over the expert weights.
    #[inline]
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.allocation.metal_buf
    }

    /// A retained (ref-counted) clone of the Metal buffer handle.
    ///
    /// Use this when the kernel dispatch outlives the `AlignedBuffer` borrow
    /// (e.g., when passing into a closure or async task).
    #[inline]
    pub fn metal_buffer_retained(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        (*self.allocation.metal_buf).clone()
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
        self.allocation.raw.as_ptr() as *mut u8
    }

    /// An immutable byte slice covering the full allocation.
    ///
    /// Safe for read-only inspection once the caller has synchronized any
    /// in-flight GPU writes that might target this buffer.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the mapping is valid and initialised (zero-filled) for
        // `size` bytes for the lifetime of `self`; only immutable access here.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size()) }
    }

    /// A mutable byte slice covering the full allocation.
    ///
    /// # Safety
    ///
    /// Same as [`Self::as_ptr`]: the GPU must not be concurrently reading or
    /// writing this buffer.
    #[inline]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is valid for `size` bytes and exclusively owned
        // by this struct at the point where the caller holds `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr(), self.size()) }
    }

    /// Copy a byte slice into the start of the buffer.
    ///
    /// This is a convenience wrapper for prefetched-expert paths that already
    /// have the expert bytes resident in CPU memory and want to preserve the
    /// aligned Metal-shared fast path for subsequent GPU dispatch.
    pub fn write_prefix(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.size() {
            return Err(MetalError::InvalidConfig(format!(
                "write_prefix: source length {} exceeds buffer size {}",
                src.len(),
                self.size()
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
        if len > self.size() {
            return Err(MetalError::InvalidConfig(format!(
                "prefix_mut: requested prefix {} exceeds buffer size {}",
                len,
                self.size()
            )));
        }

        // SAFETY: `self` is exclusively borrowed and the returned slice is
        // bounded to `len <= self.size()`.
        let bytes = unsafe { self.as_bytes_mut() };
        Ok(&mut bytes[..len])
    }

    // ── I/O ───────────────────────────────────────────────────────────────────

    /// Fill the buffer from a file using `pread(2)` at the given byte offset.
    ///
    /// Reads [`Self::size`] bytes from `fd` starting at `file_offset` directly
    /// into the aligned memory, retrying reads a signal interrupted and
    /// continuing after short reads.  Because the memory is Metal-shared, there
    /// is no copy step; the GPU can access the data immediately after this
    /// call returns.
    ///
    /// # Returns
    ///
    /// The number of bytes actually read.  This equals [`Self::size`] unless the
    /// file ends first, in which case the rest of the buffer keeps what it had
    /// (zero, for a fresh buffer; the previous contents for a reused one).
    ///
    /// # Errors
    ///
    /// Returns [`MetalError::ExecutionFailed`] if the underlying `pread` fails.
    pub fn pread(&mut self, fd: i32, file_offset: u64) -> Result<usize> {
        let count = self.size();
        self.pread_range(fd, file_offset, 0, count)
    }

    /// Fill a sub-range of the buffer from a file using `pread(2)`.
    ///
    /// Reads up to `byte_len` bytes from `fd` at `file_offset` into the buffer
    /// starting at `buf_offset`, retrying and continuing as [`Self::pread`] does.
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
        if end > self.size() {
            return Err(MetalError::InvalidConfig(format!(
                "pread_range: range [{}..{}) exceeds buffer size {}",
                buf_offset,
                end,
                self.size()
            )));
        }

        // SAFETY: `buf_offset <= end <= size`, so the pointer stays in the mapping.
        let dst = unsafe { self.as_ptr().add(buf_offset) };
        let read = read_fully(
            |at, count, offset| pread_syscall(fd, at, count, offset),
            dst,
            byte_len,
            file_offset,
        )
        .map_err(|err| {
            MetalError::ExecutionFailed(format!(
                "pread(fd={fd}, offset={file_offset}, count={byte_len}) failed: {err}"
            ))
        })?;

        trace!(
            fd,
            file_offset,
            requested = byte_len,
            read,
            "AlignedBuffer::pread"
        );
        Ok(read)
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `allocation` is taken exactly once, here.
        let allocation = unsafe { ManuallyDrop::take(&mut self.allocation) };
        match self.home.take() {
            // A pooled buffer goes back to its pool rather than shrinking it.
            Some(home) => home.give_back(AlignedBuffer {
                allocation: ManuallyDrop::new(allocation),
                home: None,
            }),
            None => drop(allocation),
        }
    }
}

// SAFETY: `AlignedBuffer` owns a mapping and a `Retained<MTLBuffer>`.
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
            .field("ptr", &self.allocation.raw.as_ptr())
            .field("size", &self.allocation.size)
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
    /// Typically the size of one expert's weight tensor, so that every expert
    /// fits in a single buffer.  The actual allocation is rounded up to a
    /// whole number of pages ([`page_size`]).
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

/// Pre-allocated pool of page-aligned, GPU-visible buffers for expert weight
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
///   GPU work has completed.  Dropping the buffer does the same.
///
/// # Memory accounting
///
/// The pool eagerly maps all `2 * K` buffers at construction time so that
/// the first `acquire` call never triggers a large allocation on the critical
/// path.  The mappings are zero-filled and committed lazily, so memory is
/// resident only for the pages an expert read touches.
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

    /// Byte capacity of each individual buffer (after page rounding).
    buffer_size: usize,
}

impl PoolInner {
    /// Put `buf` back in the queue and wake one waiter.
    fn give_back(&self, buf: AlignedBuffer) {
        debug_assert_eq!(buf.size(), self.buffer_size, "buffer from another pool");
        {
            let mut guard = self.available.lock();
            guard.push_back(buf);
            trace!(available = guard.len(), "ExpertBufferPool: buffer returned");
        }
        self.returned.notify_one();
    }

    /// Mark `buf` as checked out of this pool, so dropping it returns it.
    fn check_out(self: &Arc<Self>, mut buf: AlignedBuffer) -> AlignedBuffer {
        buf.home = Some(Arc::clone(self));
        buf
    }
}

impl ExpertBufferPool {
    /// Create a new pool and eagerly map all `2 * K` buffers.
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
        // to page rounding) from the first buffer.
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
    /// condition variable until one is returned via [`release`](Self::release)
    /// or by dropping it.
    ///
    /// # Deadlock warning
    ///
    /// Do not hold any lock that is also taken inside `release` while calling
    /// this method.
    pub fn acquire_blocking(&self) -> AlignedBuffer {
        let mut guard = self.inner.available.lock();
        loop {
            if let Some(buf) = guard.pop_front() {
                trace!(remaining = guard.len(), "ExpertBufferPool: buffer acquired");
                return self.inner.check_out(buf);
            }
            // Queue is empty — wait for a buffer to come back.
            self.inner.returned.wait(&mut guard);
        }
    }

    /// Attempt to acquire a buffer without blocking.
    ///
    /// Returns `Some(buffer)` if one is immediately available, or `None` if all
    /// buffers are currently checked out.
    pub fn try_acquire(&self) -> Option<AlignedBuffer> {
        let mut guard = self.inner.available.lock();
        let buf = guard.pop_front()?;
        trace!(
            remaining = guard.len(),
            "ExpertBufferPool: buffer acquired (try)"
        );
        Some(self.inner.check_out(buf))
    }

    /// Return a buffer to the pool.
    ///
    /// # Contract
    ///
    /// The caller **must** ensure that all GPU work using this buffer has
    /// completed before calling `release`.  The simplest way is to call
    /// `commandBuffer.waitUntilCompleted()` before releasing.
    ///
    /// Releasing is dropping: a buffer goes back to the pool it was acquired
    /// from, so a buffer from another pool returns to that pool, never to this
    /// one, and one that came from no pool is freed.
    pub fn release(&self, buf: AlignedBuffer) {
        drop(buf);
    }

    /// Total number of buffers managed by this pool (`2 * K`).
    #[inline]
    pub fn total_buffers(&self) -> usize {
        self.inner.total
    }

    /// Byte capacity of each individual buffer (after page rounding).
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    /// Total memory mapped by this pool in bytes (`total_buffers * buffer_size`).
    ///
    /// This is address space; resident memory is the pages reads have touched.
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

    const MIB: usize = 1024 * 1024;

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

    // ── read_fully ────────────────────────────────────────────────────────────

    /// A reader over `data` that returns at most `chunk` bytes per call and
    /// fails with EINTR on the calls listed in `interrupt_on`.
    fn scripted_reader<'a>(
        data: &'a [u8],
        chunk: usize,
        interrupt_on: &'a [usize],
    ) -> impl FnMut(*mut u8, usize, u64) -> std::io::Result<usize> + 'a {
        let mut calls = 0usize;
        move |dst, count, offset| {
            calls += 1;
            if interrupt_on.contains(&calls) {
                return Err(std::io::Error::from_raw_os_error(sys::EINTR));
            }
            let offset = offset as usize;
            let available = data.len().saturating_sub(offset);
            let n = count.min(chunk).min(available);
            // SAFETY: the test passes a destination with room for `count` bytes.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().add(offset), dst, n) };
            Ok(n)
        }
    }

    #[test]
    fn read_fully_continues_after_short_reads() {
        let data: Vec<u8> = (0..100u8).collect();
        let mut out = vec![0u8; 100];
        let read = read_fully(scripted_reader(&data, 7, &[]), out.as_mut_ptr(), 100, 0).unwrap();
        assert_eq!(read, 100);
        assert_eq!(out, data);
    }

    #[test]
    fn read_fully_retries_an_interrupted_call() {
        let data: Vec<u8> = (0..50u8).collect();
        let mut out = vec![0u8; 50];
        let read =
            read_fully(scripted_reader(&data, 20, &[1, 3]), out.as_mut_ptr(), 50, 0).unwrap();
        assert_eq!(read, 50);
        assert_eq!(out, data);
    }

    #[test]
    fn read_fully_stops_at_end_of_file() {
        let data: Vec<u8> = (0..10u8).collect();
        let mut out = vec![0u8; 32];
        let read = read_fully(scripted_reader(&data, 4, &[]), out.as_mut_ptr(), 32, 0).unwrap();
        assert_eq!(read, 10);
        assert_eq!(&out[..10], &data[..]);
        assert!(out[10..].iter().all(|&b| b == 0));
    }

    #[test]
    fn read_fully_honours_the_file_offset() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut out = vec![0u8; 16];
        let read = read_fully(scripted_reader(&data, 5, &[]), out.as_mut_ptr(), 16, 40).unwrap();
        assert_eq!(read, 16);
        assert_eq!(out, &data[40..56]);
    }

    #[test]
    fn read_fully_reports_a_real_error() {
        let mut out = vec![0u8; 8];
        let result = read_fully(
            |_, _, _| Err(std::io::Error::from_raw_os_error(5)),
            out.as_mut_ptr(),
            8,
            0,
        );
        assert_eq!(result.unwrap_err().raw_os_error(), Some(5));
    }

    // ── AlignedBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn test_aligned_buffer_allocation() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 4 * MIB).unwrap();

        // Allocation is a whole number of pages.
        assert_eq!(buf.size() % page_size(), 0);
        assert!(buf.size() >= 4 * MIB);

        // The raw pointer must satisfy Metal's alignment contract.
        let ptr = buf.as_ptr() as usize;
        assert_eq!(ptr % page_size(), 0, "pointer is not page-aligned");
    }

    #[test]
    fn test_aligned_buffer_size_rounding() {
        let ctx = MetalContext::new().unwrap();
        let page = page_size();

        // A 1-byte request is rounded up to exactly one page.
        let buf = AlignedBuffer::new(ctx.device(), 1).unwrap();
        assert_eq!(buf.size(), page);

        // An exact multiple stays the same.
        let buf2 = AlignedBuffer::new(ctx.device(), 3 * page).unwrap();
        assert_eq!(buf2.size(), 3 * page);

        // One byte more rounds up to the next page.
        let buf3 = AlignedBuffer::new(ctx.device(), 3 * page + 1).unwrap();
        assert_eq!(buf3.size(), 4 * page);
    }

    /// An expert smaller than a huge page costs the pool a page-rounded buffer,
    /// not a 2 MiB one.
    #[test]
    fn a_small_expert_is_not_padded_to_megabytes() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 300 * 1024).unwrap();
        assert!(buf.size() < 512 * 1024, "{} bytes", buf.size());
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
        let buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        // Verify the Metal buffer has the expected length.
        assert_eq!(buf.metal_buffer().length(), buf.size());
    }

    /// The Metal buffer aliases the mapping: the GPU sees what the CPU wrote.
    #[test]
    fn the_metal_buffer_aliases_the_allocation() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        assert_eq!(
            buf.metal_buffer().contents().as_ptr().cast_const(),
            buf.as_ptr().cast_const().cast::<std::ffi::c_void>()
        );
    }

    /// A new buffer reads as zeros.  `as_bytes` makes a `&[u8]` of the whole
    /// allocation, which must not be uninitialised memory.
    #[test]
    fn a_new_buffer_is_zero_filled() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_aligned_buffer_write_read() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();

        // Write a pattern into the aligned memory via the raw pointer.
        // SAFETY: no concurrent GPU access in this test.
        let slice = unsafe { buf.as_bytes_mut() };
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }

        // Read back and verify.
        for (i, &byte) in buf.as_bytes().iter().enumerate() {
            assert_eq!(byte, (i % 251) as u8);
        }
    }

    #[test]
    fn test_aligned_buffer_write_prefix() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        let payload = [9u8, 7, 5, 3, 1];
        buf.write_prefix(&payload).unwrap();
        assert_eq!(&buf.as_bytes()[..payload.len()], &payload);
    }

    #[test]
    fn test_pread_from_file() {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");

        // Write 4 MB of known data.
        let data: Vec<u8> = (0..4 * MIB).map(|i| (i % 199) as u8).collect();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 4 * MIB).unwrap();

        use std::os::unix::io::AsRawFd;
        let fd = tmp.as_raw_fd();
        let n = buf.pread(fd, 0).unwrap();
        assert_eq!(n, buf.size()); // entire buffer should be filled

        assert_eq!(&buf.as_bytes()[..data.len()], &data[..]);
    }

    /// A file shorter than the buffer fills a prefix and reports its length.
    #[test]
    fn pread_of_a_short_file_reports_the_length_read() {
        use std::io::Write;
        use std::os::unix::io::AsRawFd;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let data: Vec<u8> = (0..100_000).map(|i| (i % 233) as u8).collect();
        tmp.write_all(&data).unwrap();
        tmp.flush().unwrap();

        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        let n = buf.pread(tmp.as_raw_fd(), 0).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(&buf.as_bytes()[..n], &data[..]);
        assert!(buf.as_bytes()[n..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pread_range() {
        use std::io::Write;

        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let payload = b"hello expert weights";
        tmp.write_all(payload).unwrap();
        tmp.flush().unwrap();

        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();

        use std::os::unix::io::AsRawFd;
        let n = buf
            .pread_range(tmp.as_raw_fd(), 0, 0, payload.len())
            .unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(&buf.as_bytes()[..payload.len()], payload);
    }

    #[test]
    fn test_pread_range_out_of_bounds() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        // Attempt to read past the end of the buffer.
        let result = buf.pread_range(0, 0, buf.size() - 1, 2);
        assert!(result.is_err(), "out-of-bounds range should be rejected");
    }

    #[test]
    fn pread_on_a_bad_descriptor_is_an_error() {
        let ctx = MetalContext::new().unwrap();
        let mut buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        assert!(matches!(
            buf.pread(-1, 0),
            Err(MetalError::ExecutionFailed(_))
        ));
    }

    // ── ExpertBufferPool ──────────────────────────────────────────────────────

    #[test]
    fn test_pool_creation() {
        let pool = make_pool(4, 4 * MIB);
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
                buffer_size: 2 * MIB,
                k: 0,
            },
        );
        assert!(r2.is_err(), "k=0 should be rejected");
    }

    #[test]
    fn test_pool_acquire_release_cycle() {
        let pool = make_pool(2, 2 * MIB);
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

    /// A buffer that an error path or a panic drops instead of releasing must
    /// not shrink the pool; before, it was gone for good and the pool's
    /// `acquire_blocking` eventually waited forever.
    #[test]
    fn a_dropped_buffer_returns_to_the_pool() {
        let pool = make_pool(1, 2 * MIB);
        let b0 = pool.acquire_blocking();
        let b1 = pool.acquire_blocking();
        assert_eq!(pool.available_count(), 0);

        drop(b0);
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_flight_count(), 1);

        // And a blocked acquirer is woken by a drop.
        let pool = Arc::new(pool);
        let waiter = {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || {
                let first = pool.acquire_blocking();
                let second = pool.acquire_blocking(); // blocks until b1 is dropped
                drop((first, second));
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        drop(b1);
        waiter.join().expect("waiter woke and finished");
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn a_buffer_outlives_the_pool_handle_that_issued_it() {
        let pool = make_pool(1, 2 * MIB);
        let buf = pool.acquire_blocking();
        drop(pool);
        // Returning it to a pool nobody else holds is fine, and frees it.
        drop(buf);
    }

    /// Releasing a buffer to the wrong pool returns it to the pool it came from.
    #[test]
    fn a_foreign_buffer_goes_home_not_into_this_pool() {
        let ours = make_pool(1, 2 * MIB);
        let theirs = make_pool(1, 2 * MIB);
        let stray = theirs.acquire_blocking();
        ours.release(stray);
        assert_eq!(ours.available_count(), 2);
        assert_eq!(theirs.available_count(), 2);
    }

    #[test]
    fn test_pool_try_acquire_exhaustion() {
        let pool = make_pool(1, 2 * MIB); // 2 total buffers
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
        let pool = make_pool(1, 2 * MIB);
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

        let pool = StdArc::new(make_pool(1, 2 * MIB));
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
        let pool = make_pool(3, 4 * MIB);
        // 6 buffers of 4 MiB, a whole number of pages already.
        assert_eq!(pool.buffer_size(), 4 * MIB);
        assert_eq!(pool.total_bytes(), 6 * 4 * MIB);
    }

    #[test]
    fn test_pool_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExpertBufferPool>();
    }

    #[test]
    fn test_aligned_buffer_debug_format() {
        let ctx = MetalContext::new().unwrap();
        let buf = AlignedBuffer::new(ctx.device(), 2 * MIB).unwrap();
        let s = format!("{buf:?}");
        assert!(s.contains("AlignedBuffer"), "debug format: {s}");
    }

    #[test]
    fn test_pool_debug_format() {
        let pool = make_pool(2, 2 * MIB);
        let s = format!("{pool:?}");
        assert!(s.contains("ExpertBufferPool"), "debug format: {s}");
    }
}
