//! Expert I/O for SSD-offloaded MoE inference.
//!
//! Parallel `pread()` of individual experts from the packed per-layer files
//! (see [`crate::expert_layout`]) into memory the caller provides. Design:
//!
//! - **`pread()` over `mmap()`**: cold reads are several times faster (the DMA
//!   controller streams them), and no page of a file sits mapped into the
//!   process to be faulted in under the GPU's feet.
//! - **No cache of our own**: the OS page cache holds what was read recently
//!   and does it better than any cache layered over it (a hit rate near 70% at
//!   no cost to GPU memory). What is read twice is served from it.
//! - **A persistent thread pool**: workers are spawned once and live until
//!   `Drop`, so a layer pays no thread creation.
//!
//! ```text
//! ExpertOffloadContext
//!   ├── ExpertFileManager  — holds per-layer file descriptors
//!   └── ExpertIoPool       — persistent N-thread pread pool (channel-based)
//! ```

use std::fs::File;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use crate::expert_layout::ExpertPackLayout;

/// Optimal I/O thread count based on device memory bandwidth.
fn io_thread_count() -> usize {
    match pmetal_metal::context::MetalContext::global().map(|ctx| ctx.properties().device_tier) {
        Ok(pmetal_metal::context::DeviceTier::Base) => 2,
        Ok(pmetal_metal::context::DeviceTier::Pro) => 4,
        Ok(pmetal_metal::context::DeviceTier::Max) => 6,
        Ok(pmetal_metal::context::DeviceTier::Ultra) => 8,
        _ => 4, // fallback
    }
}

/// A pread task sent to a worker thread: `size` bytes at `offset` of `file`
/// into the memory at `dst_addr`.
struct ReadTask {
    file: Arc<File>,
    offset: u64,
    size: usize,
    dst_addr: usize,
    /// Index of the task in its batch, for error messages.
    index: usize,
    result_tx: mpsc::Sender<(usize, std::io::Result<()>)>,
}

/// Persistent thread pool for parallel pread() operations.
///
/// Workers are spawned once at construction and live until `Drop`.
/// Uses `std::sync::mpsc` channels for work dispatch and result collection.
/// Each worker does `FileExt::read_exact_at` (safe Rust wrapper for
/// `pread(2)`) which allows concurrent positional reads without seeking or
/// locking.
pub struct ExpertIoPool {
    workers: Vec<JoinHandle<()>>,
    task_tx: mpsc::Sender<ReadTask>,
    shutdown: Arc<AtomicBool>,
}

impl ExpertIoPool {
    /// Create a new I/O pool with a device-tier-aware number of threads.
    pub fn new() -> Self {
        Self::with_threads(io_thread_count())
    }

    /// Create with a specific number of threads.
    pub fn with_threads(num_threads: usize) -> Self {
        let (task_tx, task_rx) = mpsc::channel::<ReadTask>();
        let shutdown = Arc::new(AtomicBool::new(false));

        // Wrap task_rx in an Arc<Mutex<>> so multiple workers can share it
        let task_rx = Arc::new(Mutex::new(task_rx));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let task_rx = task_rx.clone();
            let shutdown = shutdown.clone();

            let handle = std::thread::Builder::new()
                .name("expert-io".into())
                .spawn(move || {
                    while !shutdown.load(Ordering::Relaxed) {
                        // Lock to receive one task, then release immediately
                        let task = {
                            let rx = task_rx.lock().unwrap();
                            rx.recv()
                        };
                        let Ok(task) = task else {
                            break; // Channel closed
                        };
                        let result =
                            Self::do_pread_into(&task.file, task.offset, task.size, task.dst_addr);
                        let _ = task.result_tx.send((task.index, result));
                    }
                })
                .expect("Failed to spawn expert-io thread");

            workers.push(handle);
        }

        Self {
            workers,
            task_tx,
            shutdown,
        }
    }

    /// Read bytes directly into a caller-provided mutable region.
    #[cfg(unix)]
    #[allow(unsafe_code)]
    fn do_pread_into(
        file: &File,
        offset: u64,
        size: usize,
        dst_addr: usize,
    ) -> std::io::Result<()> {
        // SAFETY: `parallel_read_into_addrs` only submits addresses its caller
        // vouches for as unique writable regions of at least `size` bytes that
        // stay alive until all results have been collected.
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_addr as *mut u8, size) };
        file.read_exact_at(dst, offset)
    }

    #[cfg(not(unix))]
    fn do_pread_into(
        _file: &File,
        _offset: u64,
        _size: usize,
        _dst_addr: usize,
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "pread is not available on this platform",
        ))
    }

    /// Read `size` bytes at each of `offsets` of `file` into the regions at
    /// `dst_addrs`, in parallel, and return once every read has finished.
    ///
    /// # Safety
    /// Each address must point to a unique writable region of at least `size`
    /// bytes that stays alive, and is not read, until this call returns.
    #[allow(unsafe_code)]
    pub unsafe fn parallel_read_into_addrs(
        &self,
        file: &Arc<File>,
        offsets: &[u64],
        dst_addrs: &[usize],
        size: usize,
    ) -> std::io::Result<()> {
        let n = offsets.len();
        if n != dst_addrs.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("offsets.len()={} != dst_addrs.len()={}", n, dst_addrs.len()),
            ));
        }
        if n == 0 {
            return Ok(());
        }
        // One read needs no hand-off.
        if n == 1 {
            return Self::do_pread_into(file, offsets[0], size, dst_addrs[0]);
        }

        let (result_tx, result_rx) = mpsc::channel();
        for (index, (&offset, &dst_addr)) in offsets.iter().zip(dst_addrs).enumerate() {
            self.task_tx
                .send(ReadTask {
                    file: file.clone(),
                    offset,
                    size,
                    dst_addr,
                    index,
                    result_tx: result_tx.clone(),
                })
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "IO pool shut down")
                })?;
        }
        drop(result_tx);

        // Every read must finish before this returns, failed or not: a worker
        // still writing into the caller's memory after an early return would be
        // a use after free.
        let mut first_error = None;
        for _ in 0..n {
            let (index, result) = result_rx.recv().map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "IO pool result channel closed",
                )
            })?;
            if let Err(e) = result
                && first_error.is_none()
            {
                first_error = Some(std::io::Error::new(
                    e.kind(),
                    format!("pread failed for task {index}: {e}"),
                ));
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// [`parallel_read_into_addrs`](Self::parallel_read_into_addrs) over
    /// caller-owned slices, which the borrow checker keeps alive and unique.
    pub fn parallel_read_into_slices(
        &self,
        file: &Arc<File>,
        offsets: &[u64],
        targets: &mut [&mut [u8]],
        size: usize,
    ) -> std::io::Result<()> {
        if offsets.len() != targets.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "offsets.len()={} != targets.len()={}",
                    offsets.len(),
                    targets.len()
                ),
            ));
        }
        for (i, target) in targets.iter().enumerate() {
            if target.len() < size {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "target {} too small for read: len={} < size={}",
                        i,
                        target.len(),
                        size
                    ),
                ));
            }
        }
        let dst_addrs: Vec<usize> = targets
            .iter_mut()
            .map(|target| target.as_mut_ptr() as usize)
            .collect();
        // SAFETY: the addresses come from unique `&mut [u8]` of at least `size`
        // bytes, borrowed for this whole call.
        #[allow(unsafe_code)]
        unsafe {
            self.parallel_read_into_addrs(file, offsets, &dst_addrs, size)
        }
    }
}

impl Default for ExpertIoPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ExpertIoPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Replace the sender with a disconnected one to close the channel.
        // This unblocks workers waiting on recv() since they'll get RecvError.
        let (dead_tx, _dead_rx) = mpsc::channel();
        let _ = std::mem::replace(&mut self.task_tx, dead_tx);
        // Now the original task_tx is dropped, closing the channel.
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl std::fmt::Debug for ExpertIoPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertIoPool")
            .field("num_workers", &self.workers.len())
            .finish()
    }
}

/// Manages open file descriptors for packed expert layer files.
#[derive(Debug)]
pub struct ExpertFileManager {
    /// Per-layer file handles (indexed by transformer layer index).
    /// None for layers without MoE.
    files: Vec<Option<Arc<File>>>,
    /// The expert pack layout.
    layout: ExpertPackLayout,
}

impl ExpertFileManager {
    /// Open all layer files for the given layout.
    pub fn open(base_dir: &Path, layout: ExpertPackLayout) -> std::io::Result<Self> {
        let mut files = Vec::with_capacity(layout.num_layers);
        for layer_idx in 0..layout.num_layers {
            if layout.moe_layer_indices.contains(&layer_idx) {
                let path = layout.layer_file_path(base_dir, layer_idx);
                let file = File::open(&path).map_err(|e| {
                    std::io::Error::new(
                        e.kind(),
                        format!("Failed to open {}: {}", path.display(), e),
                    )
                })?;
                files.push(Some(Arc::new(file)));
            } else {
                files.push(None);
            }
        }

        Ok(Self { files, layout })
    }

    /// Get the Arc file reference for a layer.
    pub fn file_for_layer(&self, layer_idx: usize) -> Option<&Arc<File>> {
        self.files.get(layer_idx).and_then(|f| f.as_ref())
    }

    /// Get the byte offset for a specific expert in a layer file.
    pub fn expert_offset(&self, expert_idx: usize) -> u64 {
        self.layout.expert_offset(expert_idx) as u64
    }

    /// Get the expert record size.
    pub fn expert_size(&self) -> usize {
        self.layout.expert_size
    }

    /// Get a reference to the layout.
    pub fn layout(&self) -> &ExpertPackLayout {
        &self.layout
    }
}

/// Complete expert offload context shared across all MoE layers.
///
/// Holds the file manager, I/O pool, and layout information needed
/// for offloaded expert inference.
#[derive(Debug)]
pub struct ExpertOffloadContext {
    /// File manager with open per-layer file descriptors.
    pub file_manager: ExpertFileManager,
    /// Persistent parallel pread I/O pool.
    pub io_pool: ExpertIoPool,
    /// Expert pack layout metadata.
    pub layout: ExpertPackLayout,
}

impl ExpertOffloadContext {
    /// Create a new offload context from a packed experts directory.
    pub fn new(packed_dir: &Path) -> std::io::Result<Self> {
        let layout = ExpertPackLayout::load(packed_dir)?;

        let file_manager = ExpertFileManager::open(packed_dir, layout.clone())?;

        let io_pool = ExpertIoPool::new();

        Ok(Self {
            file_manager,
            io_pool,
            layout,
        })
    }

    /// Read the records of `expert_indices` of `layer_idx` into the memory at
    /// `dst`, expert `i` at `dst + i * expert_size`, in parallel.
    ///
    /// # Safety
    /// `dst` must be writable for `expert_indices.len() * expert_size` bytes,
    /// and nothing else may touch them until this returns.
    #[allow(unsafe_code)]
    pub unsafe fn read_experts_into(
        &self,
        layer_idx: usize,
        expert_indices: &[usize],
        dst: *mut u8,
    ) -> std::io::Result<()> {
        let file = self.file_manager.file_for_layer(layer_idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Layer {} has no MoE expert file", layer_idx),
            )
        })?;

        let expert_size = self.file_manager.expert_size();
        let mut offsets = Vec::with_capacity(expert_indices.len());
        let mut dst_addrs = Vec::with_capacity(expert_indices.len());
        for (slot, &expert_idx) in expert_indices.iter().enumerate() {
            if expert_idx >= self.layout.num_experts {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "expert {expert_idx} out of range for {} experts",
                        self.layout.num_experts
                    ),
                ));
            }
            offsets.push(self.file_manager.expert_offset(expert_idx));
            dst_addrs.push(dst as usize + slot * expert_size);
        }
        // SAFETY: the caller vouches for `dst`; the slots are disjoint.
        unsafe {
            self.io_pool
                .parallel_read_into_addrs(file, &offsets, &dst_addrs, expert_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file_with(chunks: &[&[u8]]) -> (tempfile::TempDir, Arc<File>) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.bin");
        let mut file = File::create(&path).unwrap();
        for chunk in chunks {
            file.write_all(chunk).unwrap();
        }
        drop(file);
        (dir, Arc::new(File::open(&path).unwrap()))
    }

    #[test]
    fn reads_land_in_the_slot_their_offset_names() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
        let (_dir, file) = file_with(&[&data]);
        let pool = ExpertIoPool::new();

        let mut a = vec![0u8; 512];
        let mut b = vec![0u8; 512];
        let mut targets: Vec<&mut [u8]> = vec![a.as_mut_slice(), b.as_mut_slice()];
        pool.parallel_read_into_slices(&file, &[512, 0], &mut targets, 512)
            .unwrap();
        assert_eq!(a, &data[512..]);
        assert_eq!(b, &data[..512]);
    }

    #[test]
    fn a_single_read_and_an_empty_batch() {
        let (_dir, file) = file_with(&[&[42u8; 128]]);
        let pool = ExpertIoPool::new();
        let mut one = vec![0u8; 128];
        pool.parallel_read_into_slices(&file, &[0], &mut [one.as_mut_slice()], 128)
            .unwrap();
        assert_eq!(one, vec![42u8; 128]);
        pool.parallel_read_into_slices(&file, &[], &mut [], 0)
            .unwrap();
    }

    #[test]
    fn concurrent_callers_each_get_their_own_reads() {
        let (_dir, file) = file_with(&[&[1u8; 256], &[2u8; 256]]);
        let pool = Arc::new(ExpertIoPool::with_threads(2));

        let read = |pool: Arc<ExpertIoPool>, file: Arc<File>, offsets: [u64; 2]| {
            std::thread::spawn(move || {
                let mut a = vec![0u8; 256];
                let mut b = vec![0u8; 256];
                let mut targets: Vec<&mut [u8]> = vec![a.as_mut_slice(), b.as_mut_slice()];
                pool.parallel_read_into_slices(&file, &offsets, &mut targets, 256)
                    .unwrap();
                (a[0], b[0])
            })
        };
        let t1 = read(pool.clone(), file.clone(), [0, 256]);
        let t2 = read(pool, file, [256, 0]);
        assert_eq!(t1.join().unwrap(), (1, 2));
        assert_eq!(t2.join().unwrap(), (2, 1));
    }

    #[test]
    fn a_short_target_is_refused() {
        let file = Arc::new(tempfile::tempfile().unwrap());
        let pool = ExpertIoPool::with_threads(1);

        let mut short = vec![0u8; 8];
        let mut targets: Vec<&mut [u8]> = vec![short.as_mut_slice()];
        let err = pool
            .parallel_read_into_slices(&file, &[0], &mut targets, 16)
            .unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn a_failed_read_is_an_error_after_every_read_has_finished() {
        // One expert past the end of the file: its read fails; the others must
        // still have completed before the error is returned.
        let (_dir, file) = file_with(&[&[9u8; 256]]);
        let pool = ExpertIoPool::with_threads(2);
        let mut a = vec![0u8; 128];
        let mut b = vec![0u8; 128];
        let mut c = vec![0u8; 128];
        let mut targets: Vec<&mut [u8]> =
            vec![a.as_mut_slice(), b.as_mut_slice(), c.as_mut_slice()];
        let err = pool
            .parallel_read_into_slices(&file, &[0, 4096, 128], &mut targets, 128)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(a, vec![9u8; 128]);
        assert_eq!(c, vec![9u8; 128]);
    }
}
