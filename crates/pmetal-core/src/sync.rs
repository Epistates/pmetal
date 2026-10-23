//! Locking that survives a panic elsewhere.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Lock `mutex`, taking the data even if a thread panicked while holding it.
///
/// For state that stays usable after a panic (job tables, event buffers): one
/// failed worker must not turn every later reader into a second panic.
pub fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poisoned_lock_still_yields_its_data() {
        let mutex = Mutex::new(7);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mutex.lock().unwrap();
            panic!("poison it");
        }));
        assert!(mutex.is_poisoned());
        assert_eq!(*locked(&mutex), 7);
    }
}
