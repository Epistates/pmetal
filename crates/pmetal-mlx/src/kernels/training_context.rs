//! Training-mode flag shared by the trainers and the model forwards.
//!
//! The training loops wrap each gradient step in [`with_training_mode`], and
//! forwards read it back through [`get_training_context`] to skip paths that
//! only make sense for generation (the TurboQuant KV-cache attention, for
//! example).
//!
//! It does not choose the attention kernel. Attention that a gradient flows
//! through goes through MLX's SDPA, whose VJP is part of MLX's autodiff;
//! [`fused_sdpa`](super::fused_attention::fused_sdpa) decides that from the
//! inputs themselves, so a forward differentiated outside training mode is
//! covered too.

use pmetal_metal::MetalContext;
use std::sync::{Arc, Mutex};

use super::utils::Result;
use crate::error::MlxError;

/// Whether the current thread is running a training step.
pub struct TrainingContext {
    training: bool,
}

impl TrainingContext {
    /// Create a new training context.
    ///
    /// Fails when no Metal device is available; the training loops use that
    /// to report whether the Metal kernels can run at all.
    pub fn new() -> Result<Self> {
        MetalContext::global().map_err(|e| MlxError::Metal(e.to_string()))?;
        Ok(Self { training: false })
    }

    /// Enable training mode.
    pub fn enable_training(&mut self) {
        self.training = true;
    }

    /// Disable training mode.
    pub fn disable_training(&mut self) {
        self.training = false;
    }

    /// Check if training mode is enabled.
    pub fn is_training(&self) -> bool {
        self.training
    }
}

// Thread-local training context for easy access.
thread_local! {
    static TRAINING_CONTEXT: std::cell::RefCell<Option<Arc<Mutex<TrainingContext>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Initialize the thread's training context.
pub fn init_training_context() -> Result<Arc<Mutex<TrainingContext>>> {
    let ctx = Arc::new(Mutex::new(TrainingContext::new()?));
    TRAINING_CONTEXT.with(|c| {
        *c.borrow_mut() = Some(ctx.clone());
    });
    Ok(ctx)
}

/// Get the thread's training context.
pub fn get_training_context() -> Option<Arc<Mutex<TrainingContext>>> {
    TRAINING_CONTEXT.with(|c| c.borrow().clone())
}

/// Run `f` with training mode enabled, and disable it again afterwards.
///
/// # Example
///
/// ```ignore
/// with_training_mode(|| {
///     let logits = model.forward(&input_ids, None)?;
///     let loss = compute_loss(&logits, &labels)?;
///     Ok(loss)
/// })
/// ```
pub fn with_training_mode<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let ctx = get_training_context()
        .map(Ok)
        .unwrap_or_else(init_training_context)?;

    {
        let mut ctx_guard = ctx
            .lock()
            .map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
        ctx_guard.enable_training();
    }

    let result = f();

    {
        let mut ctx_guard = ctx
            .lock()
            .map_err(|_| MlxError::Metal("Failed to lock training context".to_string()))?;
        ctx_guard.disable_training();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_context_mode() {
        let mut ctx = TrainingContext::new().unwrap();
        assert!(!ctx.is_training());

        ctx.enable_training();
        assert!(ctx.is_training());

        ctx.disable_training();
        assert!(!ctx.is_training());
    }

    #[test]
    fn test_with_training_mode() {
        let result = with_training_mode(|| {
            let ctx = get_training_context().unwrap();
            let ctx_guard = ctx.lock().unwrap();
            assert!(ctx_guard.is_training());
            Ok(42)
        });

        assert_eq!(result.unwrap(), 42);

        if let Some(ctx) = get_training_context() {
            let ctx_guard = ctx.lock().unwrap();
            assert!(!ctx_guard.is_training());
        }
    }
}
