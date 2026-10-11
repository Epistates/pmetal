//! `bridge_extern!`, the one way this crate declares its C entry points.

/// Declares the bridge's C entry points: an `unsafe extern "C"` block, item
/// for item.
///
/// That is all it expands to in a normal build. Under the `strict-errors`
/// feature (test builds) the C functions move into a private `strict_raw`
/// module and each name becomes an `unsafe fn` of the same signature and
/// visibility that calls it and then hands control to
/// [`crate::error::strict::Call`], which panics, naming the entry point, if
/// the call raised an MLX error nothing observed. Call sites are the same in
/// both builds. One invocation per module.
macro_rules! bridge_extern {
    ($(
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;
    )*) => {
        #[cfg(not(feature = "strict-errors"))]
        unsafe extern "C" {
            $(
                $(#[$meta])*
                $vis fn $name($($arg: $ty),*) $(-> $ret)?;
            )*
        }

        #[cfg(feature = "strict-errors")]
        mod strict_raw {
            #[allow(unused_imports)]
            use super::*;

            unsafe extern "C" {
                $(
                    pub(super) fn $name($($arg: $ty),*) $(-> $ret)?;
                )*
            }
        }

        $(
            #[cfg(feature = "strict-errors")]
            $(#[$meta])*
            #[inline]
            #[track_caller]
            #[allow(dead_code, clippy::too_many_arguments)]
            $vis unsafe fn $name($($arg: $ty),*) $(-> $ret)? {
                let call = $crate::error::strict::Call::begin();
                // SAFETY: the caller upholds the C function's contract, as it
                // would calling it directly.
                let out = unsafe { strict_raw::$name($($arg),*) };
                call.end(stringify!($name));
                out
            }
        )*
    };
}
