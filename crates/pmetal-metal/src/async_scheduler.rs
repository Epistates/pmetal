//! Async command buffer scheduling for optimized GPU-CPU synchronization.
//!
//! This module provides infrastructure for asynchronous GPU execution that
//! eliminates unnecessary CPU-GPU synchronization points.
//!
//! # The Problem
//!
//! Traditional Metal execution:
//! ```text
//! CPU: [encode]──[wait]──[encode]──[wait]──[encode]──[wait]
//! GPU:          [exec]            [exec]            [exec]
//! ```
//!
//! GPU is idle while CPU waits. With ~50 kernel dispatches per inference
//! and ~0.1ms overhead per sync, this adds 5ms+ latency.
//!
//! # The Solution
//!
//! Async execution with double buffering:
//! ```text
//! CPU: [encode A]──[encode B]──[encode A]──[encode B]
//! GPU:             [exec A]────[exec B]────[exec A]────[exec B]
//! ```
//!
//! CPU prepares next batch while GPU executes current batch.
//!
//! # Components
//!
//! - [`AsyncScheduler`]: Central coordinator for async command buffer management
//! - [`InFlightBuffer`]: Tracks command buffers with completion status
//! - [`DoubleBuffer`]: Two-buffer system for overlapping compute
//! - [`TripleBuffer`]: Three-buffer system for maximum throughput
//!
//! # Performance Targets
//!
//! - Reduce sync overhead from ~5ms to <0.5ms per training step
//! - Enable 40%+ throughput improvement via overlapping execution
//! - Support for batched operations across multiple kernel dispatches

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCommandEncoder,
};
use parking_lot::RwLock;

use crate::context::MetalContext;
use crate::error::{MetalError, Result};

// =============================================================================
// GPU Completion Token Trait
// =============================================================================

/// Trait for tracking GPU command buffer completion.
///
/// This provides a unified interface for all completion token types,
/// allowing generic code to work with any completion tracking mechanism.
///
/// # Error Handling
///
/// The `wait_checked()` method propagates GPU errors, allowing callers to
/// detect and handle execution failures (shader errors, resource issues, etc.).
///
/// # Example
///
/// ```ignore
/// fn wait_for_gpu<T: GpuCompletionToken>(token: &T) -> Result<()> {
///     token.wait_checked()
/// }
/// ```
pub trait GpuCompletionToken: Send + Sync {
    /// Wait for the GPU work to complete (blocking).
    ///
    /// This is the simple version that doesn't return errors.
    /// Use `wait_checked()` if you need error propagation.
    fn wait(&self);

    /// Wait for completion and return any GPU errors.
    ///
    /// This is the preferred method when you need to detect GPU execution failures.
    fn wait_checked(&self) -> Result<()>;

    /// Check if the GPU work has completed.
    ///
    /// Returns `true` if completed (successfully or with error), `false` if still running.
    fn is_complete(&self) -> bool;

    /// Wait with timeout.
    ///
    /// Returns `Ok(true)` if completed within timeout.
    /// Returns `Ok(false)` if timeout expired (work still in progress).
    /// Returns `Err` if GPU execution failed.
    fn wait_timeout(&self, timeout: Duration) -> Result<bool>;

    /// Get the operation ID for tracking/debugging.
    fn operation_id(&self) -> u64;

    /// Check if execution resulted in an error.
    ///
    /// Returns `None` if still running or completed successfully.
    /// Returns `Some(error)` if execution failed.
    fn error(&self) -> Option<String>;
}

// =============================================================================
// Async Scheduler
// =============================================================================

/// Central coordinator for async command buffer management.
///
/// The scheduler maintains a pool of command buffers and tracks their
/// lifecycle to enable overlapping CPU-GPU execution.
///
/// # Thread Safety
///
/// The scheduler is thread-safe and can be shared across threads.
/// Command buffers are dispatched to a single command queue (Metal
/// guarantees FIFO ordering within a queue).
pub struct AsyncScheduler {
    ctx: Arc<MetalContext>,
    /// Maximum pool size (for tracking, actual pooling is simplified).
    max_pool_size: usize,
    /// Counter for tracking operations.
    operation_counter: AtomicU64,
    /// Statistics for monitoring.
    stats: RwLock<SchedulerStats>,
}

/// Statistics for monitoring scheduler performance.
#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    /// Total command buffers created.
    pub buffers_created: u64,
    /// Total command buffers committed.
    pub commits: u64,
    /// Total sync waits performed.
    pub sync_waits: u64,
    /// Total async dispatches.
    pub async_dispatches: u64,
}

impl AsyncScheduler {
    /// Create a new async scheduler.
    ///
    /// # Arguments
    /// * `ctx` - Metal context
    /// * `max_pool_size` - Maximum number of command buffers to track
    pub fn new(ctx: Arc<MetalContext>, max_pool_size: usize) -> Result<Self> {
        Ok(Self {
            ctx,
            max_pool_size,
            operation_counter: AtomicU64::new(0),
            stats: RwLock::new(SchedulerStats::default()),
        })
    }

    /// Get the Metal context.
    #[inline]
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get current scheduler statistics.
    pub fn stats(&self) -> SchedulerStats {
        self.stats.read().clone()
    }

    /// Create a new command buffer.
    fn create_command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let buffer = self
            .ctx
            .command_queue()
            .commandBuffer()
            .ok_or(MetalError::CommandBufferCreation)?;

        self.stats.write().buffers_created += 1;
        Ok(buffer)
    }

    /// Create an in-flight buffer wrapper for async tracking.
    pub fn create_in_flight(&self) -> Result<InFlightBuffer> {
        let command_buffer = self.create_command_buffer()?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::EncoderCreation)?;

        Ok(InFlightBuffer::new(command_buffer, encoder))
    }

    /// Commit an in-flight buffer asynchronously.
    ///
    /// Returns a completion token that can be used to wait for completion.
    pub fn commit_async(&self, mut buffer: InFlightBuffer) -> Result<CompletionToken> {
        buffer.end_encoding();
        let command_buffer = buffer.take_command_buffer()?;

        let operation_id = self.operation_counter.fetch_add(1, Ordering::SeqCst);

        command_buffer.commit();
        self.stats.write().async_dispatches += 1;

        Ok(CompletionToken::new(command_buffer, operation_id))
    }

    /// Commit and wait for completion (blocking).
    ///
    /// Use this when you need results immediately.
    pub fn commit_sync(&self, mut buffer: InFlightBuffer) -> Result<()> {
        buffer.end_encoding();
        let command_buffer = buffer.take_command_buffer()?;

        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        self.stats.write().sync_waits += 1;
        self.stats.write().commits += 1;

        if let Some(error) = command_buffer.error() {
            return Err(MetalError::ExecutionFailed(error.to_string()));
        }

        Ok(())
    }
}

// SAFETY: AsyncScheduler is thread-safe - all mutable state is protected
// by atomic operations or locks, and Metal objects are thread-safe.
unsafe impl Send for AsyncScheduler {}
unsafe impl Sync for AsyncScheduler {}

// =============================================================================
// In-Flight Buffer
// =============================================================================

/// A command buffer that is being encoded or in-flight.
///
/// This wrapper tracks the encoding state and provides methods
/// for adding compute dispatches.
pub struct InFlightBuffer {
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    dispatch_count: usize,
    encoding_ended: bool,
}

impl InFlightBuffer {
    /// Create a new in-flight buffer.
    fn new(
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    ) -> Self {
        Self {
            command_buffer: Some(command_buffer),
            encoder: Some(encoder),
            dispatch_count: 0,
            encoding_ended: false,
        }
    }

    /// Get the encoder for adding dispatches.
    pub fn encoder(&self) -> Option<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder.as_ref().map(|e| e.as_ref())
    }

    /// Get mutable encoder reference.
    pub fn encoder_mut(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder
            .as_ref()
            .map(|e| e.as_ref())
            .ok_or(MetalError::EncoderCreation)
    }

    /// Get the number of dispatches queued.
    #[inline]
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    /// Increment dispatch count after adding a dispatch.
    #[inline]
    pub fn add_dispatch(&mut self) {
        self.dispatch_count += 1;
    }

    /// End encoding (called before commit).
    fn end_encoding(&mut self) {
        if !self.encoding_ended {
            if let Some(encoder) = self.encoder.take() {
                encoder.endEncoding();
            }
            self.encoding_ended = true;
        }
    }

    /// Take the command buffer (consumes self).
    fn take_command_buffer(
        &mut self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.command_buffer
            .take()
            .ok_or(MetalError::CommandBufferCreation)
    }
}

impl Drop for InFlightBuffer {
    fn drop(&mut self) {
        // Ensure encoding is properly ended
        self.end_encoding();
    }
}

// =============================================================================
// Completion Token
// =============================================================================

/// Token for tracking async command buffer completion.
///
/// Use this to wait for GPU work to complete or check completion status.
/// Implements [`GpuCompletionToken`] for use in generic contexts.
pub struct CompletionToken {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    operation_id: u64,
}

impl CompletionToken {
    /// Create a new completion token (internal use).
    pub(crate) fn new(
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        operation_id: u64,
    ) -> Self {
        Self {
            command_buffer,
            operation_id,
        }
    }
}

impl GpuCompletionToken for CompletionToken {
    fn wait(&self) {
        self.command_buffer.waitUntilCompleted();
    }

    fn wait_checked(&self) -> Result<()> {
        self.command_buffer.waitUntilCompleted();
        if let Some(err) = self.command_buffer.error() {
            return Err(MetalError::ExecutionFailed(err.to_string()));
        }
        Ok(())
    }

    #[inline]
    fn is_complete(&self) -> bool {
        let status = self.command_buffer.status();
        status == MTLCommandBufferStatus::Completed || status == MTLCommandBufferStatus::Error
    }

    fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        use std::time::Instant;

        const POLL_INTERVAL: Duration = Duration::from_micros(100);
        let deadline = Instant::now() + timeout;

        while !self.is_complete() {
            let now = Instant::now();
            if now >= deadline {
                return Ok(false);
            }
            // Sleep for poll interval or remaining time, whichever is smaller
            let remaining = deadline - now;
            std::thread::sleep(POLL_INTERVAL.min(remaining));
        }

        // Check for GPU errors after completion
        if let Some(err) = self.command_buffer.error() {
            return Err(MetalError::ExecutionFailed(err.to_string()));
        }
        Ok(true)
    }

    #[inline]
    fn operation_id(&self) -> u64 {
        self.operation_id
    }

    fn error(&self) -> Option<String> {
        self.command_buffer.error().map(|e| e.to_string())
    }
}

// SAFETY: CompletionToken wraps Metal objects which are thread-safe.
unsafe impl Send for CompletionToken {}
unsafe impl Sync for CompletionToken {}

// =============================================================================
// Double Buffer
// =============================================================================

/// Double-buffer system for overlapping CPU-GPU execution.
///
/// While GPU executes buffer A, CPU prepares buffer B.
/// Then they swap: GPU executes B while CPU prepares A.
///
/// # GPU Hang Detection
///
/// The `acquire()` method includes timeout-based hang detection. If the previous
/// GPU command buffer doesn't complete within the configured timeout (default 30 seconds),
/// an error is returned.
///
/// # Example
///
/// ```ignore
/// let mut double_buf = DoubleBuffer::new(scheduler)?;
///
/// for batch in batches {
///     // Get buffer for encoding
///     let buffer = double_buf.acquire()?;
///
///     // Encode work
///     encode_kernels(&mut buffer, &batch)?;
///
///     // Submit and swap
///     double_buf.submit(buffer)?;
/// }
///
/// // Wait for final work
/// double_buf.synchronize();
/// ```
pub struct DoubleBuffer {
    scheduler: Arc<AsyncScheduler>,
    /// Currently executing completion token (if any).
    in_flight: Option<CompletionToken>,
    /// Buffer index for tracking.
    buffer_index: usize,
    /// Timeout for GPU hang detection.
    gpu_timeout: Duration,
}

impl DoubleBuffer {
    /// Create a new double buffer system with default timeout.
    pub fn new(scheduler: Arc<AsyncScheduler>) -> Self {
        Self {
            scheduler,
            in_flight: None,
            buffer_index: 0,
            gpu_timeout: DEFAULT_GPU_TIMEOUT,
        }
    }

    /// Create with custom timeout for GPU hang detection.
    pub fn with_timeout(scheduler: Arc<AsyncScheduler>, gpu_timeout: Duration) -> Self {
        Self {
            scheduler,
            in_flight: None,
            buffer_index: 0,
            gpu_timeout,
        }
    }

    /// Set the GPU timeout for hang detection.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.gpu_timeout = timeout;
    }

    /// Acquire a buffer for encoding.
    ///
    /// If a previous buffer is still in-flight, waits for it with timeout.
    /// Returns an error if the GPU appears hung (timeout exceeded) or if
    /// a GPU execution error occurred.
    pub fn acquire(&mut self) -> Result<InFlightBuffer> {
        // Wait for previous in-flight work to complete with timeout
        if let Some(token) = self.in_flight.take() {
            match token.wait_timeout(self.gpu_timeout) {
                Ok(true) => {
                    // Completed successfully
                }
                Ok(false) => {
                    return Err(MetalError::GpuTimeout {
                        operation_id: token.operation_id(),
                        timeout: self.gpu_timeout,
                    });
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        self.scheduler.create_in_flight()
    }

    /// Submit a buffer for async execution.
    pub fn submit(&mut self, buffer: InFlightBuffer) -> Result<()> {
        let token = self.scheduler.commit_async(buffer)?;
        self.in_flight = Some(token);
        self.buffer_index += 1;
        Ok(())
    }

    /// Wait for all in-flight work to complete (no timeout).
    pub fn synchronize(&mut self) {
        if let Some(token) = self.in_flight.take() {
            token.wait();
        }
    }

    /// Wait for in-flight work with timeout-based hang detection.
    pub fn synchronize_checked(&mut self) -> Result<()> {
        if let Some(token) = self.in_flight.take() {
            match token.wait_timeout(self.gpu_timeout) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(MetalError::GpuTimeout {
                        operation_id: token.operation_id(),
                        timeout: self.gpu_timeout,
                    });
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Get the current buffer index.
    #[inline]
    pub fn buffer_index(&self) -> usize {
        self.buffer_index
    }
}

// =============================================================================
// Triple Buffer
// =============================================================================

/// Default timeout for GPU hang detection (30 seconds).
///
/// If a GPU command buffer takes longer than this to complete, it's considered hung.
pub const DEFAULT_GPU_TIMEOUT: Duration = Duration::from_secs(30);

/// Triple-buffer system for maximum throughput.
///
/// Maintains up to 3 buffers in flight:
/// - Buffer A: GPU executing
/// - Buffer B: GPU queued
/// - Buffer C: CPU encoding
///
/// This provides maximum overlap but uses more memory.
///
/// # GPU Hang Detection
///
/// The `acquire()` method includes timeout-based hang detection. If a GPU command
/// buffer doesn't complete within the configured timeout (default 30 seconds),
/// an error is returned. This prevents the system from freezing indefinitely
/// when the GPU encounters issues.
pub struct TripleBuffer {
    scheduler: Arc<AsyncScheduler>,
    /// Queue of in-flight completion tokens.
    in_flight: VecDeque<CompletionToken>,
    /// Maximum number of in-flight buffers.
    max_in_flight: usize,
    /// Buffer index for tracking.
    buffer_index: usize,
    /// Timeout for GPU hang detection.
    gpu_timeout: Duration,
}

impl TripleBuffer {
    /// Create a new triple buffer system with default timeout.
    pub fn new(scheduler: Arc<AsyncScheduler>) -> Self {
        Self {
            scheduler,
            in_flight: VecDeque::with_capacity(3),
            max_in_flight: 3,
            buffer_index: 0,
            gpu_timeout: DEFAULT_GPU_TIMEOUT,
        }
    }

    /// Create with custom max in-flight count and default timeout.
    pub fn with_max_in_flight(scheduler: Arc<AsyncScheduler>, max: usize) -> Self {
        Self {
            scheduler,
            in_flight: VecDeque::with_capacity(max),
            max_in_flight: max,
            buffer_index: 0,
            gpu_timeout: DEFAULT_GPU_TIMEOUT,
        }
    }

    /// Create with custom timeout for GPU hang detection.
    ///
    /// # Arguments
    /// * `scheduler` - The async scheduler
    /// * `max_in_flight` - Maximum concurrent buffers
    /// * `gpu_timeout` - Timeout for GPU command buffer completion
    pub fn with_timeout(
        scheduler: Arc<AsyncScheduler>,
        max_in_flight: usize,
        gpu_timeout: Duration,
    ) -> Self {
        Self {
            scheduler,
            in_flight: VecDeque::with_capacity(max_in_flight),
            max_in_flight,
            buffer_index: 0,
            gpu_timeout,
        }
    }

    /// Set the GPU timeout for hang detection.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.gpu_timeout = timeout;
    }

    /// Get the current GPU timeout setting.
    pub fn timeout(&self) -> Duration {
        self.gpu_timeout
    }

    /// Acquire a buffer for encoding.
    ///
    /// If max in-flight buffers are reached, waits for oldest to complete.
    /// Returns an error if the GPU appears hung (timeout exceeded) or if
    /// a GPU execution error occurred.
    ///
    /// # Errors
    ///
    /// - `MetalError::GpuTimeout` - GPU command buffer didn't complete within timeout
    /// - `MetalError::ExecutionFailed` - GPU execution failed with an error
    pub fn acquire(&mut self) -> Result<InFlightBuffer> {
        // If at max capacity, wait for oldest to complete with timeout
        while self.in_flight.len() >= self.max_in_flight {
            if let Some(token) = self.in_flight.pop_front() {
                // Wait with timeout for GPU hang detection
                match token.wait_timeout(self.gpu_timeout) {
                    Ok(true) => {
                        // Completed successfully
                    }
                    Ok(false) => {
                        // Timeout expired - GPU appears hung
                        return Err(MetalError::GpuTimeout {
                            operation_id: token.operation_id(),
                            timeout: self.gpu_timeout,
                        });
                    }
                    Err(e) => {
                        // GPU execution error
                        return Err(e);
                    }
                }
            }
        }

        self.scheduler.create_in_flight()
    }

    /// Submit a buffer for async execution.
    pub fn submit(&mut self, buffer: InFlightBuffer) -> Result<()> {
        let token = self.scheduler.commit_async(buffer)?;
        self.in_flight.push_back(token);
        self.buffer_index += 1;
        Ok(())
    }

    /// Wait for all in-flight work to complete.
    ///
    /// This version doesn't use timeout - use `synchronize_with_timeout()` for hang detection.
    pub fn synchronize(&mut self) {
        while let Some(token) = self.in_flight.pop_front() {
            token.wait();
        }
    }

    /// Wait for all in-flight work with timeout-based hang detection.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer times out or has a GPU execution error.
    pub fn synchronize_checked(&mut self) -> Result<()> {
        while let Some(token) = self.in_flight.pop_front() {
            match token.wait_timeout(self.gpu_timeout) {
                Ok(true) => {
                    // Completed successfully
                }
                Ok(false) => {
                    return Err(MetalError::GpuTimeout {
                        operation_id: token.operation_id(),
                        timeout: self.gpu_timeout,
                    });
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Get the number of currently in-flight buffers.
    #[inline]
    pub fn in_flight_count(&self) -> usize {
        // Count only tokens that haven't completed
        self.in_flight.iter().filter(|t| !t.is_complete()).count()
    }

    /// Get the current buffer index.
    #[inline]
    pub fn buffer_index(&self) -> usize {
        self.buffer_index
    }
}

// =============================================================================
// Async Batch Builder
// =============================================================================

/// Builder for constructing batched async operations.
///
/// This provides a fluent interface for encoding multiple dispatches
/// into a single command buffer.
pub struct AsyncBatchBuilder<'a> {
    scheduler: &'a AsyncScheduler,
    buffer: InFlightBuffer,
}

impl<'a> AsyncBatchBuilder<'a> {
    /// Create a new batch builder.
    pub fn new(scheduler: &'a AsyncScheduler) -> Result<Self> {
        let buffer = scheduler.create_in_flight()?;
        Ok(Self { scheduler, buffer })
    }

    /// Get the encoder for adding dispatches.
    pub fn encoder(&mut self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.buffer.encoder_mut()
    }

    /// Add a dispatch and increment count.
    pub fn add_dispatch(&mut self) {
        self.buffer.add_dispatch();
    }

    /// Get current dispatch count.
    #[inline]
    pub fn dispatch_count(&self) -> usize {
        self.buffer.dispatch_count()
    }

    /// Execute synchronously (blocking).
    pub fn execute_sync(self) -> Result<()> {
        self.scheduler.commit_sync(self.buffer)
    }

    /// Execute asynchronously.
    pub fn execute_async(self) -> Result<CompletionToken> {
        self.scheduler.commit_async(self.buffer)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_scheduler() -> Arc<AsyncScheduler> {
        let ctx = Arc::new(MetalContext::new().expect("Failed to create Metal context"));
        Arc::new(
            AsyncScheduler::new(ctx, 4).expect("Failed to create scheduler"),
        )
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = create_test_scheduler();
        let stats = scheduler.stats();
        assert_eq!(stats.buffers_created, 0);
    }

    #[test]
    fn test_in_flight_buffer_creation() {
        let scheduler = create_test_scheduler();
        let buffer = scheduler.create_in_flight().unwrap();
        assert_eq!(buffer.dispatch_count(), 0);
        assert!(buffer.encoder().is_some());
    }

    #[test]
    fn test_sync_execution() {
        let scheduler = create_test_scheduler();
        let buffer = scheduler.create_in_flight().unwrap();

        // Execute (empty buffer, just tests the flow)
        scheduler.commit_sync(buffer).unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.sync_waits, 1);
        assert_eq!(stats.commits, 1);
    }

    #[test]
    fn test_async_execution() {
        let scheduler = create_test_scheduler();
        let buffer = scheduler.create_in_flight().unwrap();

        // Execute async
        let token = scheduler.commit_async(buffer).unwrap();

        // Wait for completion
        token.wait();
        assert!(token.is_complete());

        let stats = scheduler.stats();
        assert_eq!(stats.async_dispatches, 1);
    }

    #[test]
    fn test_double_buffer() {
        let scheduler = create_test_scheduler();
        let mut double_buf = DoubleBuffer::new(scheduler.clone());

        // Submit a few buffers
        for _ in 0..3 {
            let buffer = double_buf.acquire().unwrap();
            double_buf.submit(buffer).unwrap();
        }

        double_buf.synchronize();
        assert_eq!(double_buf.buffer_index(), 3);
    }

    #[test]
    fn test_triple_buffer() {
        let scheduler = create_test_scheduler();
        let mut triple_buf = TripleBuffer::new(scheduler.clone());

        // Submit several buffers
        for _ in 0..5 {
            let buffer = triple_buf.acquire().unwrap();
            triple_buf.submit(buffer).unwrap();
        }

        triple_buf.synchronize();
        assert_eq!(triple_buf.buffer_index(), 5);
    }

    #[test]
    fn test_async_batch_builder() {
        let scheduler = create_test_scheduler();
        let builder = AsyncBatchBuilder::new(&scheduler).unwrap();

        // Just test the builder flow
        builder.execute_sync().unwrap();
    }

    #[test]
    fn test_completion_token_timeout() {
        let scheduler = create_test_scheduler();
        let buffer = scheduler.create_in_flight().unwrap();

        let token = scheduler.commit_async(buffer).unwrap();

        // Should complete quickly (empty buffer)
        let completed = token.wait_timeout(std::time::Duration::from_secs(1)).unwrap();
        assert!(completed);
    }

    #[test]
    fn test_gpu_completion_token_trait() {
        // Test that the trait works with generic code
        fn wait_for_completion<T: GpuCompletionToken>(token: &T) -> Result<()> {
            token.wait_checked()
        }

        let scheduler = create_test_scheduler();
        let buffer = scheduler.create_in_flight().unwrap();
        let token = scheduler.commit_async(buffer).unwrap();

        // Use via trait
        wait_for_completion(&token).unwrap();
        assert!(token.is_complete());
        assert!(token.error().is_none());
    }

    #[test]
    fn test_multiple_async_dispatches() {
        let scheduler = create_test_scheduler();

        // Submit multiple async buffers
        let mut tokens = Vec::new();
        for _ in 0..3 {
            let buffer = scheduler.create_in_flight().unwrap();
            tokens.push(scheduler.commit_async(buffer).unwrap());
        }

        // Wait for all
        for token in tokens {
            token.wait();
            assert!(token.is_complete());
        }

        let stats = scheduler.stats();
        assert_eq!(stats.async_dispatches, 3);
    }
}
