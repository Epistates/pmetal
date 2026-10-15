//! Training-mode flag shared by the trainers and the model forwards.
//!
//! The training loops wrap each gradient step in [`with_training_mode`], and
//! forwards read it back through [`is_training`] to skip paths that only make
//! sense for generation (the TurboQuant KV-cache attention, for example).
//!
//! It does not choose the attention kernel. Attention that a gradient flows
//! through goes through MLX's SDPA, whose VJP is part of MLX's autodiff;
//! [`fused_sdpa`](super::fused_attention::fused_sdpa) decides that from the
//! inputs themselves, so a forward differentiated outside training mode is
//! covered too.

use std::cell::Cell;

use pmetal_runtime::compat::Exception;

thread_local! {
    static TRAINING: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is inside [`with_training_mode`].
pub fn is_training() -> bool {
    TRAINING.with(Cell::get)
}

/// Run `f` with training mode on, and put the mode back as it was afterwards
/// (also if `f` panics), so calls nest.
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
pub fn with_training_mode<T>(f: impl FnOnce() -> Result<T, Exception>) -> Result<T, Exception> {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            TRAINING.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TRAINING.with(|flag| flag.replace(true)));
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_is_on_inside_and_restored_after() {
        assert!(!is_training());
        let seen = with_training_mode(|| Ok(is_training())).unwrap();
        assert!(seen);
        assert!(!is_training());
    }

    #[test]
    fn calls_nest_and_an_error_restores_the_mode() {
        with_training_mode(|| {
            let _ = with_training_mode(|| Err::<(), _>(Exception::custom("boom")));
            assert!(is_training(), "the inner call left the outer mode on");
            Ok(())
        })
        .unwrap();
        assert!(!is_training());
    }

    #[test]
    fn a_panic_restores_the_mode() {
        let result = std::panic::catch_unwind(|| {
            let _ = with_training_mode::<()>(|| panic!("inside"));
        });
        assert!(result.is_err());
        assert!(!is_training());
    }

    #[test]
    fn the_mode_is_per_thread() {
        with_training_mode(|| {
            assert!(!std::thread::spawn(is_training).join().unwrap());
            Ok(())
        })
        .unwrap();
    }
}
