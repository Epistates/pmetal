//! MLX distributed: process groups and collective operations.
//!
//! Whichever backends MLX was built with (ring over TCP, JACCL over
//! Thunderbolt RDMA, MPI) are configured by MLX's own environment variables
//! before launch (`MLX_RANK`, `MLX_HOSTFILE`, `MLX_IBV_DEVICES`, ...).
//! [`Group::init`] joins the group they describe. Every collective is lazy,
//! like any other op, and communicates when it is evaluated. Passing no group
//! uses MLX's default, the one `init` made.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use crate::InlineArray;
use crate::error::{BridgeError, BridgeResult, check_last_error};
use crate::inline_array::{RawBuf, from_raw_buf};

unsafe extern "C" {
    fn mlx_inline_distributed_is_available() -> bool;
    fn mlx_inline_distributed_init(strict: bool) -> *mut c_void;
    fn mlx_inline_distributed_group_split(
        group: *const c_void,
        color: i32,
        key: i32,
    ) -> *mut c_void;
    fn mlx_inline_distributed_group_rank(group: *const c_void) -> i32;
    fn mlx_inline_distributed_group_size(group: *const c_void) -> i32;
    fn mlx_inline_distributed_group_free(group: *mut c_void);
    fn mlx_inline_distributed_all_sum(dst: *mut RawBuf, x: *const RawBuf, group: *const c_void);
    fn mlx_inline_distributed_all_gather(dst: *mut RawBuf, x: *const RawBuf, group: *const c_void);
    fn mlx_inline_distributed_all_max(dst: *mut RawBuf, x: *const RawBuf, group: *const c_void);
    fn mlx_inline_distributed_all_min(dst: *mut RawBuf, x: *const RawBuf, group: *const c_void);
    fn mlx_inline_distributed_sum_scatter(dst: *mut RawBuf, x: *const RawBuf, group: *const c_void);
    fn mlx_inline_distributed_send(
        dst: *mut RawBuf,
        x: *const RawBuf,
        to: i32,
        group: *const c_void,
    );
    fn mlx_inline_distributed_recv(
        dst: *mut RawBuf,
        shape: *const i32,
        ndim: usize,
        dtype: i32,
        from: i32,
        group: *const c_void,
    );
    fn mlx_inline_distributed_recv_like(
        dst: *mut RawBuf,
        x: *const RawBuf,
        from: i32,
        group: *const c_void,
    );
}

/// Whether MLX was built with any distributed backend.
pub fn is_available() -> bool {
    // SAFETY: no arguments; the C side catches every exception.
    unsafe { mlx_inline_distributed_is_available() }
}

/// A group of processes that communicate.
pub struct Group(NonNull<c_void>);

// SAFETY: the handle owns an `mlx::core::distributed::Group`, a shared_ptr to
// the backend's group, which MLX shares across threads; every operation on it
// is a const method or a collective that synchronizes through the backend.
unsafe impl Send for Group {}
unsafe impl Sync for Group {}

impl Group {
    /// Join the group MLX's environment describes. With `strict`, failing to
    /// set a backend up is an error; without, a process launched with no
    /// distributed configuration gets a group of one, whose collectives
    /// return their input.
    pub fn init(strict: bool) -> BridgeResult<Self> {
        // SAFETY: the C side catches every exception and returns null.
        Self::from_raw(unsafe { mlx_inline_distributed_init(strict) }, "init")
    }

    /// This process's rank in the group, from 0.
    pub fn rank(&self) -> i32 {
        // SAFETY: `self.0` is a live group handle.
        unsafe { mlx_inline_distributed_group_rank(self.0.as_ptr()) }
    }

    /// The number of processes in the group.
    pub fn size(&self) -> i32 {
        // SAFETY: `self.0` is a live group handle.
        unsafe { mlx_inline_distributed_group_size(self.0.as_ptr()) }
    }

    /// Split into sub-groups: processes passing the same `color` share one,
    /// ranked by `key` (a negative key keeps the current order).
    pub fn split(&self, color: i32, key: i32) -> BridgeResult<Self> {
        // SAFETY: `self.0` is a live group handle; the C side catches every
        // exception and returns null.
        Self::from_raw(
            unsafe { mlx_inline_distributed_group_split(self.0.as_ptr(), color, key) },
            "split",
        )
    }

    fn from_raw(raw: *mut c_void, what: &str) -> BridgeResult<Self> {
        match NonNull::new(raw) {
            Some(handle) => Ok(Self(handle)),
            None => {
                check_last_error()?;
                Err(BridgeError::Unknown(format!(
                    "distributed group {what} returned no group"
                )))
            }
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        // SAFETY: the handle came from init or split and is freed only here.
        unsafe { mlx_inline_distributed_group_free(self.0.as_ptr()) }
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Group(rank={}, size={})", self.rank(), self.size())
    }
}

fn group_ptr(group: Option<&Group>) -> *const c_void {
    group.map_or(std::ptr::null(), |g| g.0.as_ptr().cast_const())
}

/// Build one array through `op`, which placement-news it into the buffer it
/// is given (or a placeholder on error), and surface a C++ exception.
fn checked(op: impl FnOnce(*mut RawBuf)) -> BridgeResult<InlineArray> {
    let mut dst = MaybeUninit::<RawBuf>::uninit();
    op(dst.as_mut_ptr());
    // SAFETY: every `mlx_inline_distributed_*` op initialises `dst`, with
    // the result or, on an exception, a placeholder.
    let out = unsafe { from_raw_buf(dst.assume_init()) };
    check_last_error()?;
    Ok(out)
}

/// Element-wise sum over the group.
pub fn all_sum(x: &InlineArray, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: valid array and group handles; `dst` is written by the callee.
    checked(|dst| unsafe { mlx_inline_distributed_all_sum(dst, &x.raw, group_ptr(group)) })
}

/// Every rank's `x` concatenated along the first axis, rank 0 first.
pub fn all_gather(x: &InlineArray, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_all_gather(dst, &x.raw, group_ptr(group)) })
}

/// Element-wise maximum over the group.
pub fn all_max(x: &InlineArray, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_all_max(dst, &x.raw, group_ptr(group)) })
}

/// Element-wise minimum over the group.
pub fn all_min(x: &InlineArray, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_all_min(dst, &x.raw, group_ptr(group)) })
}

/// Sum over the group, each rank keeping its slice of the first axis.
pub fn sum_scatter(x: &InlineArray, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_sum_scatter(dst, &x.raw, group_ptr(group)) })
}

/// Send `x` to rank `to`. Evaluating the returned array sends it.
pub fn send(x: &InlineArray, to: i32, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_send(dst, &x.raw, to, group_ptr(group)) })
}

/// Receive an array of `shape` and `dtype` (a bridge dtype code) from rank
/// `from`.
pub fn recv(
    shape: &[i32],
    dtype: i32,
    from: i32,
    group: Option<&Group>,
) -> BridgeResult<InlineArray> {
    // SAFETY: `shape` is valid for `shape.len()` reads; otherwise as in
    // `all_sum`.
    checked(|dst| unsafe {
        mlx_inline_distributed_recv(
            dst,
            shape.as_ptr(),
            shape.len(),
            dtype,
            from,
            group_ptr(group),
        )
    })
}

/// Receive an array shaped and typed like `x` from rank `from`.
pub fn recv_like(x: &InlineArray, from: i32, group: Option<&Group>) -> BridgeResult<InlineArray> {
    // SAFETY: as in `all_sum`.
    checked(|dst| unsafe { mlx_inline_distributed_recv_like(dst, &x.raw, from, group_ptr(group)) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A process launched without a distributed configuration is a group of
    /// one, and every collective hands back its input.
    #[test]
    fn a_lone_process_is_a_group_of_one() {
        let group = Group::init(false).expect("non-strict init");
        assert_eq!((group.rank(), group.size()), (0, 1));

        let x = InlineArray::from_f32_slice(&[1.0, -2.0, 3.0], &[3]);
        for (name, out) in [
            ("all_sum", all_sum(&x, Some(&group))),
            ("all_max", all_max(&x, Some(&group))),
            ("all_min", all_min(&x, None)),
            ("all_gather", all_gather(&x, None)),
            ("sum_scatter", sum_scatter(&x, Some(&group))),
        ] {
            let mut out = out.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                out.to_f32_vec(3).expect("eval"),
                vec![1.0, -2.0, 3.0],
                "{name}"
            );
        }
        check_last_error().expect("no bridge op failed");
    }
}
