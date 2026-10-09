//! Stopping a run at the step that went wrong.
//!
//! A bridge op that throws inside a gradient trace does not unwind: the
//! bridge records the exception on the thread, hands back an empty array, and
//! the step goes on to report a NaN loss and zero gradients. A loop that only
//! logs the loss then carries on, every later step with it, and at the end
//! saves an adapter that no step moved. Each trainer reads its step's loss
//! back through [`check_step`] instead, which ends the run there, naming the
//! step and the op.

use pmetal_bridge::compat::Exception;

/// The step's loss, or why the run has to stop at it.
///
/// `step` is the 1-based number of the step that produced `loss`. Any
/// exception a bridge op recorded since the last check fails it first, since
/// that is the cause and a non-finite loss only the symptom; a thrown op can
/// also leave a finite loss behind. The op that threw is rarely the step's
/// last, and every op that succeeds clears the last-op error slot, so this
/// reads the one that holds the first unobserved error instead.
pub fn check_step(step: usize, loss: f32) -> Result<f32, Exception> {
    if let Err(e) = pmetal_bridge::check_unobserved_error() {
        return Err(Exception::custom(format!(
            "training step {step} failed: {e}"
        )));
    }
    if !loss.is_finite() {
        return Err(Exception::custom(format!(
            "training step {step} produced a loss of {loss}; stopping rather than \
             training on it"
        )));
    }
    Ok(loss)
}

/// [`check_step`] for a gradient norm, which a finite loss can still come
/// with a non-finite one of.
pub fn check_grad_norm(step: usize, norm: f32) -> Result<f32, Exception> {
    if !norm.is_finite() {
        return Err(Exception::custom(format!(
            "training step {step} produced a gradient norm of {norm}; stopping rather \
             than applying it"
        )));
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::{Array, ops};

    #[test]
    fn a_finite_loss_passes() {
        assert_eq!(check_step(3, 1.25).unwrap(), 1.25);
        assert_eq!(check_grad_norm(3, 0.5).unwrap(), 0.5);
    }

    #[test]
    fn a_non_finite_loss_names_the_step() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = check_step(7, bad).unwrap_err().to_string();
            assert!(err.contains("step 7"), "{err}");
        }
        let err = check_grad_norm(2, f32::NAN).unwrap_err().to_string();
        assert!(err.contains("step 2"), "{err}");
    }

    /// An op that threw is reported, with its message, even though the loss
    /// read back from it looks fine.
    #[test]
    fn a_recorded_bridge_error_fails_the_step() {
        // Mismatched inner dimensions: the bridge records the exception.
        let _ = ops::matmul(&Array::ones_f32(&[2, 3]), &Array::ones_f32(&[4, 5]));
        let err = check_step(4, 0.5).unwrap_err().to_string();
        assert!(err.contains("step 4"), "{err}");
        assert!(err.contains("matmul"), "{err}");
        // Drained: the next step starts clean.
        assert!(check_step(5, 0.5).is_ok());
    }
}
