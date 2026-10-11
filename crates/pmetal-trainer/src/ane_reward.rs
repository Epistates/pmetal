//! Asynchronous reward model scoring for GRPO training.
//!
//! Enables pipelined execution where reward scoring overlaps with the next
//! training step.  On Apple Silicon, the GPU training forward/backward pass
//! and the reward model's inference run on the same unified memory bus but
//! compete for GPU compute.  By moving reward scoring to a background thread
//! (and optionally to the ANE when a small enough reward model is available),
//! the GPU is free to continue training while scores for the previous batch
//! are being computed.
//!
//! # Pipeline Architecture
//!
//! ```text
//! Step N:   [Generate] ──► [Submit score (async)] ──► [Train]
//! Step N+1:               [Generate]              ──► [Collect score + Submit] ──► [Train]
//!                                                      ^^^
//!                                              overlaps with step N score
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use pmetal_trainer::ane_reward::{AsyncRewardModel, PendingRewards};
//! use pmetal_trainer::{CombinedReward, RewardFunction};
//!
//! let inner = CombinedReward::new();
//! let async_model = AsyncRewardModel::new(Box::new(inner));
//!
//! let prompts = vec!["Hello".to_string()];
//! let completions = vec!["World".to_string()];
//! let pending = async_model
//!     .score_async(prompts, completions)
//!     .expect("failed to submit");
//!
//! let rewards = pending.collect().expect("scoring failed");
//! ```

use std::sync::mpsc;
use std::thread;

use pmetal_bridge::compat::Array;

use crate::grpo::{GrpoError, GrpoResult, RewardFunction};

// ---------------------------------------------------------------------------
// Internal channel types
// ---------------------------------------------------------------------------

/// A scoring request sent to the background worker thread.
struct ScoreRequest {
    prompts: Vec<String>,
    completions: Vec<String>,
    /// One-shot channel: the worker sends the result back here.
    response_tx: mpsc::SyncSender<GrpoResult<Vec<f64>>>,
}

// ---------------------------------------------------------------------------
// PendingRewards — handle for an in-flight scoring request
// ---------------------------------------------------------------------------

/// A handle to a reward scoring request that is executing in the background.
///
/// Call [`PendingRewards::collect`] to block until results are available.
/// If the background thread panicked or was dropped, `collect` returns an error
/// rather than panicking the caller.
pub struct PendingRewards {
    rx: mpsc::Receiver<GrpoResult<Vec<f64>>>,
}

impl PendingRewards {
    /// Block the current thread until the scoring results are ready.
    ///
    /// Returns the reward scores in the same order as the completions
    /// originally passed to [`AsyncRewardModel::score_async`].
    ///
    /// # Errors
    ///
    /// Returns [`GrpoError::Reward`] if:
    /// - The background worker thread panicked.
    /// - The scoring computation itself failed.
    pub fn collect(self) -> GrpoResult<Vec<f64>> {
        self.rx.recv().map_err(|_| {
            GrpoError::Reward(
                "Reward scoring worker thread terminated unexpectedly (channel closed)".into(),
            )
        })?
    }

    /// Non-blocking check: returns `Some(result)` if scores are already ready,
    /// or `None` if the worker is still computing.
    ///
    /// Useful for polling in time-sensitive loops; prefer `collect` when
    /// you are ready to synchronize.
    pub fn try_collect(&self) -> Option<GrpoResult<Vec<f64>>> {
        match self.rx.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err(GrpoError::Reward(
                "Reward scoring worker thread terminated unexpectedly".into(),
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// AsyncRewardModel
// ---------------------------------------------------------------------------

/// Asynchronous wrapper around any [`RewardFunction`].
///
/// Spawns a single background thread that processes scoring requests from a
/// channel.  The caller submits a batch via `score_async` and receives a
/// [`PendingRewards`] handle that can be collected later, after the GPU has
/// made progress on the next training step.
///
/// # Thread Safety
///
/// The inner reward function must be `Send + Sync`.  The background thread
/// owns exclusive access to the inner function, so no locking is required
/// inside the worker loop.
///
/// # Backpressure
///
/// The request channel is bounded to one slot (`SyncSender` with capacity 1).
/// If the caller submits a second request before the first has been collected,
/// `score_async` will block until the worker is ready.  This prevents
/// unbounded queue growth when scoring is slower than generation.
///
/// # Shutdown
///
/// The worker thread exits cleanly when this struct is dropped (the sender end
/// of the channel is closed, causing the worker's `recv()` to return an error
/// and the thread to exit).
pub struct AsyncRewardModel {
    /// Sender half of the bounded scoring-request channel.
    request_tx: mpsc::SyncSender<ScoreRequest>,
    /// The worker thread handle.  Kept alive to detect panics via `join`.
    ///
    /// We do not join on drop because it could block the training loop.
    /// The thread exits when the `SyncSender` is dropped (see `Drop` section).
    _worker: thread::JoinHandle<()>,
    /// Descriptive name surfaced by `RewardFunction::name`.
    name: String,
}

impl AsyncRewardModel {
    /// Wrap any `RewardFunction` for asynchronous, pipelined scoring.
    ///
    /// Spawns a background thread immediately.  The thread lives until this
    /// `AsyncRewardModel` is dropped.
    ///
    /// # Arguments
    ///
    /// * `inner` — the underlying reward function (e.g. `MLRewardModel`,
    ///   `CombinedReward`, or any custom implementation).
    pub fn new(inner: Box<dyn RewardFunction>) -> Self {
        let name = format!("async_{}", inner.name());

        // Bounded channel with capacity 1 to provide natural backpressure:
        // the caller cannot submit a new request until the previous one has
        // been dequeued by the worker.
        let (request_tx, request_rx) = mpsc::sync_channel::<ScoreRequest>(1);

        let worker = thread::Builder::new()
            .name("pmetal-reward-scorer".into())
            .spawn(move || {
                // Worker loop: process requests until the sender side is dropped.
                while let Ok(req) = request_rx.recv() {
                    let result = inner.compute(&req.prompts, &req.completions, None);
                    // Ignore send errors: the caller may have dropped the
                    // receiver if it timed out or exited early.
                    let _ = req.response_tx.send(result);
                }
                tracing::debug!("AsyncRewardModel worker thread exiting");
            })
            .expect("failed to spawn reward scorer thread");

        Self {
            request_tx,
            _worker: worker,
            name,
        }
    }

    /// Submit a batch of completions for asynchronous scoring.
    ///
    /// Returns immediately with a [`PendingRewards`] handle.  The actual
    /// scoring runs on the background thread concurrently with whatever the
    /// caller does next (typically a GPU training step).
    ///
    /// If the bounded channel is full (i.e., the previous request has not yet
    /// been dequeued by the worker), this call blocks until capacity is
    /// available.
    ///
    /// # Arguments
    ///
    /// * `prompts` — prompt strings, one per completion.
    /// * `completions` — completion strings to score.
    ///
    /// # Errors
    ///
    /// Returns [`GrpoError::Reward`] if the background worker thread has
    /// terminated unexpectedly.
    pub fn score_async(
        &self,
        prompts: Vec<String>,
        completions: Vec<String>,
    ) -> GrpoResult<PendingRewards> {
        // One-shot channel for this specific request's response.
        // SyncSender with capacity 1: the worker sends exactly one result.
        let (response_tx, response_rx) = mpsc::sync_channel(1);

        let req = ScoreRequest {
            prompts,
            completions,
            response_tx,
        };

        // This will block if the bounded request channel is at capacity
        // (i.e., the worker hasn't dequeued the previous request yet).
        self.request_tx.send(req).map_err(|_| {
            GrpoError::Reward(
                "Failed to send scoring request: reward worker thread has terminated".into(),
            )
        })?;

        Ok(PendingRewards { rx: response_rx })
    }
}

impl RewardFunction for AsyncRewardModel {
    /// Synchronous scoring via the background thread (submit + block).
    ///
    /// This path is provided for API compatibility with `CombinedReward` and
    /// other callers that expect synchronous `RewardFunction::compute`.
    ///
    /// For maximum throughput in a training loop, prefer the explicit
    /// `score_async` + `PendingRewards::collect` pattern instead.
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        _images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let pending = self.score_async(prompts.to_vec(), completions.to_vec())?;
        pending.collect()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// AsyncRewardModel is Send+Sync because:
// - `request_tx: SyncSender<ScoreRequest>` is Send (std guarantees).
// - `_worker: JoinHandle<()>` is Send.
// - `name: String` is Send+Sync.
// The inner `Box<dyn RewardFunction>` is Send+Sync and lives only in the
// worker thread, never shared with callers after construction.
//
// No explicit unsafe impl needed since all constituent types satisfy the bounds.

// ---------------------------------------------------------------------------
// PipelinedGrpoSession — stateful pipelined scoring handle
// ---------------------------------------------------------------------------

/// State machine for a single pipelined GRPO training iteration.
///
/// Tracks whether a scoring request from the *previous* step is still in
/// flight and provides a clean interface for the two-phase pipelined loop:
///
/// 1. `begin_step(prompts, completions)` — submit scoring for step N and
///    return any pending results from step N-1.
/// 2. `end_step()` — collect scores for step N (blocks until ready).
///
/// This separation gives the GPU training forward/backward pass the full
/// duration of step N to overlap with the ANE/CPU reward scoring from step N.
pub struct PipelinedGrpoSession<'a> {
    model: &'a AsyncRewardModel,
    /// In-flight scoring request from the previous step, if any.
    pending: Option<PendingRewards>,
}

impl<'a> PipelinedGrpoSession<'a> {
    /// Create a new pipelined session bound to the given async reward model.
    pub fn new(model: &'a AsyncRewardModel) -> Self {
        Self {
            model,
            pending: None,
        }
    }

    /// Begin a new training step by submitting a scoring request.
    ///
    /// Returns the rewards from the **previous** step (or `None` if this is
    /// the first step).  The current step's scores are submitted in the
    /// background and can be collected by calling `end_step`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`AsyncRewardModel::score_async`] or from
    /// collecting the previous step's results.
    pub fn begin_step(
        &mut self,
        prompts: Vec<String>,
        completions: Vec<String>,
    ) -> GrpoResult<Option<Vec<f64>>> {
        // Collect results from the previous step before we submit the new request
        // (the bounded channel has capacity 1 and would block if we submitted first).
        let prev_rewards = self.pending.take().map(|p| p.collect()).transpose()?;

        // Submit the new scoring request for this step.
        let new_pending = self.model.score_async(prompts, completions)?;
        self.pending = Some(new_pending);

        Ok(prev_rewards)
    }

    /// Collect the scores for the most recently submitted step.
    ///
    /// Blocks until scoring is complete.  Should be called once per step,
    /// after the training forward/backward pass has been submitted to the GPU.
    ///
    /// Returns `None` if no request is in flight (i.e., `begin_step` was
    /// never called).
    ///
    /// # Errors
    ///
    /// Propagates scoring errors from the background thread.
    pub fn end_step(&mut self) -> GrpoResult<Option<Vec<f64>>> {
        self.pending.take().map(|p| p.collect()).transpose()
    }

    /// Drain any remaining in-flight request (e.g., at end of training).
    ///
    /// Equivalent to `end_step` but named for clarity at training loop exit.
    pub fn flush(&mut self) -> GrpoResult<Option<Vec<f64>>> {
        self.end_step()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpo::RewardFunction;

    /// Trivial reward function that returns the length of each completion.
    struct LenReward;
    impl RewardFunction for LenReward {
        fn compute(
            &self,
            _prompts: &[String],
            completions: &[String],
            _images: Option<&[Vec<pmetal_bridge::compat::Array>]>,
        ) -> GrpoResult<Vec<f64>> {
            Ok(completions.iter().map(|c| c.len() as f64).collect())
        }
        fn name(&self) -> &str {
            "len_reward"
        }
    }

    /// Reward function that always errors, used to test error propagation.
    struct ErrorReward;
    impl RewardFunction for ErrorReward {
        fn compute(
            &self,
            _: &[String],
            _: &[String],
            _: Option<&[Vec<pmetal_bridge::compat::Array>]>,
        ) -> GrpoResult<Vec<f64>> {
            Err(GrpoError::Reward("intentional test error".into()))
        }
        fn name(&self) -> &str {
            "error_reward"
        }
    }

    // -----------------------------------------------------------------------
    // Basic async scoring
    // -----------------------------------------------------------------------

    #[test]
    fn test_score_async_basic() {
        let model = AsyncRewardModel::new(Box::new(LenReward));

        let prompts = vec!["p".into(), "p".into()];
        let completions = vec!["hello".into(), "world!".into()];

        let pending = model.score_async(prompts, completions).unwrap();
        let rewards = pending.collect().unwrap();

        assert_eq!(rewards.len(), 2);
        assert!(
            (rewards[0] - 5.0).abs() < 1e-10,
            "expected len=5, got {}",
            rewards[0]
        );
        assert!(
            (rewards[1] - 6.0).abs() < 1e-10,
            "expected len=6, got {}",
            rewards[1]
        );
    }

    // -----------------------------------------------------------------------
    // Synchronous fallback via RewardFunction trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_sync_compute_via_trait() {
        let model = AsyncRewardModel::new(Box::new(LenReward));

        let prompts = vec!["p".into()];
        let completions = vec!["abc".into()];
        let rewards = model.compute(&prompts, &completions, None).unwrap();

        assert_eq!(rewards, vec![3.0]);
    }

    // -----------------------------------------------------------------------
    // Name propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_name_propagation() {
        let model = AsyncRewardModel::new(Box::new(LenReward));
        assert_eq!(model.name(), "async_len_reward");
    }

    // -----------------------------------------------------------------------
    // Error propagation from inner reward function
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_propagation() {
        let model = AsyncRewardModel::new(Box::new(ErrorReward));

        let pending = model
            .score_async(vec!["p".into()], vec!["c".into()])
            .unwrap();
        let result = pending.collect();

        assert!(result.is_err(), "expected error from ErrorReward");
        match result.unwrap_err() {
            GrpoError::Reward(msg) => assert!(msg.contains("intentional test error")),
            other => panic!("unexpected error variant: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Multiple sequential requests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_sequential_requests() {
        let model = AsyncRewardModel::new(Box::new(LenReward));

        for i in 1..=5usize {
            let text = "x".repeat(i);
            let rewards = model
                .compute(&["p".into()], std::slice::from_ref(&text), None)
                .unwrap();
            assert!(
                (rewards[0] - i as f64).abs() < 1e-10,
                "step {i}: expected {i}, got {}",
                rewards[0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // PipelinedGrpoSession — two-step pipeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipelined_session_basic() {
        let model = AsyncRewardModel::new(Box::new(LenReward));
        let mut session = PipelinedGrpoSession::new(&model);

        // Step 1: no previous rewards.
        let prev = session
            .begin_step(vec!["p".into()], vec!["hello".into()])
            .unwrap();
        assert!(prev.is_none(), "first step should have no previous rewards");

        // Simulate GPU training work here (no-op in test).

        // Collect step 1 rewards.
        let step1 = session.end_step().unwrap();
        assert_eq!(step1, Some(vec![5.0]));

        // Step 2: begin_step collects step 1 and submits step 2.
        let prev2 = session
            .begin_step(vec!["p".into()], vec!["ab".into()])
            .unwrap();
        // prev2 is None because we already called end_step after step 1.
        assert!(prev2.is_none());

        let step2 = session.end_step().unwrap();
        assert_eq!(step2, Some(vec![2.0]));
    }

    // -----------------------------------------------------------------------
    // PipelinedGrpoSession — true overlap (begin_step before collecting)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipelined_session_overlap() {
        let model = AsyncRewardModel::new(Box::new(LenReward));
        let mut session = PipelinedGrpoSession::new(&model);

        // Step 1: submit without collecting first.
        let _ = session
            .begin_step(vec!["p".into()], vec!["abc".into()])
            .unwrap();

        // Step 2: begin_step drains step 1's result, submits step 2.
        let prev = session
            .begin_step(vec!["p".into()], vec!["de".into()])
            .unwrap();
        assert_eq!(prev, Some(vec![3.0]), "step 1 reward (len=3)");

        let step2 = session.end_step().unwrap();
        assert_eq!(step2, Some(vec![2.0]), "step 2 reward (len=2)");
    }

    // -----------------------------------------------------------------------
    // PipelinedGrpoSession — flush at end
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipelined_session_flush() {
        let model = AsyncRewardModel::new(Box::new(LenReward));
        let mut session = PipelinedGrpoSession::new(&model);

        let _ = session
            .begin_step(vec!["p".into()], vec!["flushed".into()])
            .unwrap();

        let result = session.flush().unwrap();
        assert_eq!(result, Some(vec![7.0]));

        // Double-flush should return None (nothing pending).
        let empty = session.flush().unwrap();
        assert!(empty.is_none());
    }

    // -----------------------------------------------------------------------
    // try_collect — non-blocking poll
    // -----------------------------------------------------------------------

    #[test]
    fn test_try_collect_eventually_ready() {
        let model = AsyncRewardModel::new(Box::new(LenReward));
        let pending = model
            .score_async(vec!["p".into()], vec!["xyz".into()])
            .unwrap();

        // Poll until ready — use sleep instead of spin_loop to give the
        // async task time to complete, especially on CI with fewer cores.
        let mut result = None;
        for _ in 0..100 {
            if let Some(r) = pending.try_collect() {
                result = Some(r);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let rewards = result
            .expect("try_collect never returned a result")
            .unwrap();
        assert_eq!(rewards, vec![3.0]);
    }
}
