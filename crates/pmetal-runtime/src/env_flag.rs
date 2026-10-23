//! Environment switches read once.
//!
//! A forward pass or a decode step runs thousands of times a second, and
//! `std::env::var_os` takes the process-wide environment lock and walks the
//! environment on every call. A diagnostic switch is read at its first use and
//! kept: changing it afterwards has no effect, which is what a switch set in
//! the shell that starts the process needs.

/// Whether the environment variable `$name` is set, read on first use and
/// cached for the life of the process.
macro_rules! env_flag {
    ($name:literal) => {{
        static FLAG: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *FLAG.get_or_init(|| ::std::env::var_os($name).is_some())
    }};
}
pub(crate) use env_flag;

#[cfg(test)]
mod tests {

    fn probe() -> bool {
        env_flag!("PMETAL_TEST_ENV_FLAG_READ_ONCE")
    }

    /// The variable is read at the first call; a change after it is not seen.
    #[test]
    fn the_variable_is_read_once() {
        const NAME: &str = "PMETAL_TEST_ENV_FLAG_READ_ONCE";
        assert!(std::env::var_os(NAME).is_none(), "unset {NAME}");
        assert!(!probe());
        // SAFETY: no other test reads or writes this variable.
        unsafe { std::env::set_var(NAME, "1") };
        assert!(!probe(), "the flag was read again");
        // SAFETY: as above.
        unsafe { std::env::remove_var(NAME) };
    }
}
