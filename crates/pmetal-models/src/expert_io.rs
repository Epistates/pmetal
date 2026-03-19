//! Expert I/O infrastructure for SSD-offloaded MoE inference.
//!
//! Implements high-performance parallel `pread()` I/O for loading individual expert
//! weights from packed binary files. Key design decisions from flash-moe:
//!
//! - **pread() over mmap()**: 5x faster for cold reads (DMA controller optimization)
//! - **No custom caching**: OS page cache provides 38% better performance
//! - **Persistent thread pool**: Workers spawned once, live until Drop — eliminates
//!   thread creation overhead per layer (was thread::scope per call)
//!
//! # Architecture
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

/// Number of I/O threads for parallel pread.
/// 4 threads gives 3-4x speedup for top-k=4-8 expert reads.
const NUM_IO_THREADS: usize = 4;

/// A pread task sent to a worker thread.
struct IoWork {
    /// Byte offset in the file.
    offset: u64,
    /// Number of bytes to read.
    size: usize,
    /// Index into the results vec (for ordering).
    result_idx: usize,
}

/// Result from a worker thread.
struct IoResult {
    /// Index into the results vec.
    result_idx: usize,
    /// The data read, or an error.
    data: std::io::Result<Vec<u8>>,
}

/// Persistent thread pool for parallel pread() operations.
///
/// Workers are spawned once at construction and live until `Drop`.
/// Uses `std::sync::mpsc` channels for work dispatch and result collection.
/// Each worker does `FileExt::read_at()` (safe Rust wrapper for `pread(2)`)
/// which allows concurrent positional reads without seeking or locking.
pub struct ExpertIoPool {
    workers: Vec<JoinHandle<()>>,
    task_tx: mpsc::Sender<(IoWork, Arc<File>)>,
    result_rx: mpsc::Receiver<IoResult>,
    shutdown: Arc<AtomicBool>,
}

impl ExpertIoPool {
    /// Create a new I/O pool with the default number of threads.
    pub fn new() -> Self {
        Self::with_threads(NUM_IO_THREADS)
    }

    /// Create with a specific number of threads.
    pub fn with_threads(num_threads: usize) -> Self {
        let (task_tx, task_rx) = mpsc::channel::<(IoWork, Arc<File>)>();
        let (result_tx, result_rx) = mpsc::channel::<IoResult>();
        let shutdown = Arc::new(AtomicBool::new(false));

        // Wrap task_rx in an Arc<Mutex<>> so multiple workers can share it
        let task_rx = Arc::new(Mutex::new(task_rx));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
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

                        match task {
                            Ok((work, file)) => {
                                let data = Self::do_pread(&file, work.offset, work.size);
                                let _ = result_tx.send(IoResult {
                                    result_idx: work.result_idx,
                                    data,
                                });
                            }
                            Err(_) => break, // Channel closed
                        }
                    }
                })
                .expect("Failed to spawn expert-io thread");

            workers.push(handle);
        }

        Self {
            workers,
            task_tx,
            result_rx,
            shutdown,
        }
    }

    /// Read bytes at a given offset using pread (no seek, no lock).
    #[cfg(unix)]
    fn do_pread(file: &File, offset: u64, size: usize) -> std::io::Result<Vec<u8>> {
        let mut buf = vec![0u8; size];
        file.read_exact_at(&mut buf, offset)?;
        Ok(buf)
    }

    #[cfg(not(unix))]
    fn do_pread(_file: &File, _offset: u64, _size: usize) -> std::io::Result<Vec<u8>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "pread is not available on this platform",
        ))
    }

    /// Read multiple expert chunks from a file in parallel.
    ///
    /// Dispatches `offsets.len()` pread tasks to the worker pool and collects
    /// results in order. Each task reads `size` bytes at the given offset.
    ///
    /// Returns one `Vec<u8>` per offset, in the same order as the input.
    pub fn parallel_read(
        &self,
        file: &Arc<File>,
        offsets: &[u64],
        size: usize,
    ) -> std::io::Result<Vec<Vec<u8>>> {
        let n = offsets.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // For a single task, skip the channel overhead
        if n == 1 {
            let data = Self::do_pread(file, offsets[0], size)?;
            return Ok(vec![data]);
        }

        // Dispatch all tasks
        for (i, &offset) in offsets.iter().enumerate() {
            let work = IoWork {
                offset,
                size,
                result_idx: i,
            };
            self.task_tx
                .send((work, file.clone()))
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "IO pool shut down")
                })?;
        }

        // Collect all results
        let mut results: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        for _ in 0..n {
            let result = self.result_rx.recv().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "IO pool result channel closed")
            })?;
            match result.data {
                Ok(data) => {
                    results[result.result_idx] = Some(data);
                }
                Err(e) => {
                    return Err(std::io::Error::new(
                        e.kind(),
                        format!("pread failed for task {}: {}", result.result_idx, e),
                    ));
                }
            }
        }

        // Unwrap all results (all should be Some at this point)
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
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
        // Drop the sender to unblock workers waiting on recv()
        // (We can't drop self.task_tx directly, but when ExpertIoPool is dropped,
        // the sender is dropped automatically, closing the channel)
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

    /// Read `k` expert weight buffers for a given layer in parallel.
    ///
    /// Returns `k` byte vectors, each containing the raw expert weight data.
    pub fn read_experts(
        &self,
        layer_idx: usize,
        expert_indices: &[usize],
    ) -> std::io::Result<Vec<Vec<u8>>> {
        let file = self.file_manager.file_for_layer(layer_idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Layer {} has no MoE expert file", layer_idx),
            )
        })?;

        let expert_size = self.file_manager.expert_size();
        let offsets: Vec<u64> = expert_indices
            .iter()
            .map(|&idx| self.file_manager.expert_offset(idx))
            .collect();

        self.io_pool.parallel_read(file, &offsets, expert_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_io_pool_basic() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.bin");

        // Write test data
        let mut file = File::create(&path).unwrap();
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        file.write_all(&data).unwrap();
        drop(file);

        // Read back with pool
        let file = Arc::new(File::open(&path).unwrap());

        let pool = ExpertIoPool::new();
        let offsets = vec![0u64, 512];
        let results = pool.parallel_read(&file, &offsets, 512).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 512);
        assert_eq!(results[1].len(), 512);
        assert_eq!(results[0][0], 0);
        assert_eq!(results[1][0], 0); // 512 % 256 = 0
    }

    #[test]
    fn test_io_pool_single_task() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.bin");

        let mut file = File::create(&path).unwrap();
        file.write_all(&[42u8; 128]).unwrap();
        drop(file);

        let file = Arc::new(File::open(&path).unwrap());
        let pool = ExpertIoPool::new();
        let results = pool.parallel_read(&file, &[0], 128).unwrap();
        assert_eq!(results[0], vec![42u8; 128]);
    }

    #[test]
    fn test_io_pool_empty() {
        let pool = ExpertIoPool::new();
        let file = Arc::new(tempfile::tempfile().unwrap());
        let results = pool.parallel_read(&file, &[], 0).unwrap();
        assert!(results.is_empty());
    }
}
