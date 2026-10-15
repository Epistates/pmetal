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

bridge_extern! {
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

/// Bytes of gradients one collective carries by default: large enough that a
/// ring over a fast link is bandwidth-bound, small enough to start sending
/// before the last gradient is ready.
pub const DEFAULT_BUCKET_BYTES: usize = 32 << 20;

/// Average `grads` over `group` in place, as data-parallel training needs
/// after each rank's backward pass.
///
/// The gradients stay on the device and the collectives stay lazy: they are
/// flattened, packed into buckets of up to `bucket_bytes` (one dtype to a
/// bucket, the dtype the gradients have, so each bucket is one `all_sum` over
/// a single contiguous array rather than one per parameter), summed over the
/// group, divided by its size and split back to their shapes. A group of one
/// leaves the gradients as they are.
pub fn average_gradients<K>(
    grads: &mut std::collections::HashMap<K, InlineArray>,
    group: Option<&Group>,
    bucket_bytes: usize,
) -> BridgeResult<()>
where
    K: Eq + std::hash::Hash + Ord + Clone,
{
    let size = group.map_or_else(|| Group::init(false).map(|g| g.size()), |g| Ok(g.size()))?;
    if size <= 1 {
        return Ok(());
    }
    average_gradients_with(grads, size, bucket_bytes, |bucket| all_sum(bucket, group))
}

/// [`average_gradients`] with the collective given: `reduce` sums a flat
/// bucket over `size` ranks.
fn average_gradients_with<K>(
    grads: &mut std::collections::HashMap<K, InlineArray>,
    size: i32,
    bucket_bytes: usize,
    reduce: impl Fn(&InlineArray) -> BridgeResult<InlineArray>,
) -> BridgeResult<()>
where
    K: Eq + std::hash::Hash + Ord + Clone,
{
    // Every rank packs the same buckets in the same order: by dtype, then key.
    let mut keys: Vec<K> = grads.keys().cloned().collect();
    keys.sort();
    let mut buckets: Vec<(i32, Vec<K>, usize)> = Vec::new();
    for key in keys {
        let grad = &grads[&key];
        let dtype = grad.dtype_raw();
        let bytes = grad.nbytes();
        match buckets.last_mut() {
            Some((d, members, held))
                if *d == dtype && (*held == 0 || *held + bytes <= bucket_bytes) =>
            {
                members.push(key);
                *held += bytes;
            }
            _ => buckets.push((dtype, vec![key], bytes)),
        }
    }

    let mean = 1.0 / size as f32;
    for (_, members, _) in buckets {
        let flat: Vec<InlineArray> = members.iter().map(|k| grads[k].reshape(&[-1])).collect();
        let packed = crate::compat::ops::concatenate_owned_axis(&flat, 0);
        let averaged = reduce(&packed)?.mul_scalar(mean);
        let mut offsets = Vec::with_capacity(members.len());
        let mut at = 0;
        for k in &members[..members.len() - 1] {
            at += grads[k].size() as i32;
            offsets.push(at);
        }
        let pieces = if offsets.is_empty() {
            vec![averaged]
        } else {
            averaged.split(&offsets, 0)
        };
        for (k, piece) in members.iter().zip(pieces) {
            let shape = grads[k].shape().to_vec();
            grads.insert(k.clone(), piece.reshape(&shape));
        }
    }
    check_last_error()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::check_unobserved_error;

    /// Packing, summing and splitting returns every gradient to its own shape
    /// and place, across dtypes and several buckets. Three ranks hold `g`, `2g`
    /// and `3g`, so the sum is `6g` and the average `2g`.
    #[test]
    fn averaging_gradients_unpacks_every_bucket_to_its_parameter() {
        use crate::compat::Dtype;
        use std::collections::HashMap;
        let f32_ = Dtype::Float32.as_i32();
        let bf16 = Dtype::Bfloat16.as_i32();
        let make = |values: &[f32], shape: &[i32], dtype: i32| {
            InlineArray::from_f32_slice(values, shape).as_dtype(dtype)
        };
        let mut grads: HashMap<String, InlineArray> = HashMap::from([
            (
                "a".into(),
                make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], f32_),
            ),
            ("b".into(), make(&[-1.0, 0.5], &[2], f32_)),
            ("c".into(), make(&[8.0, 16.0], &[1, 2], bf16)),
            ("d".into(), make(&[3.0], &[1], f32_)),
        ]);
        let want: Vec<(&str, Vec<f32>)> = vec![
            ("a", vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
            ("b", vec![-2.0, 1.0]),
            ("c", vec![16.0, 32.0]),
            ("d", vec![6.0]),
        ];
        // A bucket holds 24 bytes: `a` alone, then `b` with nothing after it
        // (`c` is another dtype), and so on.
        average_gradients_with(&mut grads, 3, 24, |bucket| Ok(bucket.mul_scalar(6.0)))
            .expect("average");
        for (key, values) in want {
            let mut g = grads[key].as_dtype(f32_);
            assert_eq!(g.to_f32_vec(values.len()).expect("eval"), values, "{key}");
        }
        assert_eq!(grads["a"].shape(), &[2, 3]);
        assert_eq!(grads["c"].shape(), &[1, 2]);
        assert_eq!(grads["c"].dtype(), Dtype::Bfloat16);
        check_unobserved_error().expect("no bridge op failed");
    }

    #[test]
    fn a_group_of_one_leaves_gradients_alone() {
        let group = Group::init(false).expect("init");
        let mut grads = std::collections::HashMap::from([(
            "w".to_string(),
            InlineArray::from_f32_slice(&[1.0, 2.0], &[2]),
        )]);
        average_gradients(&mut grads, Some(&group), DEFAULT_BUCKET_BYTES).expect("average");
        let mut w = grads["w"].clone();
        assert_eq!(w.to_f32_vec(2).expect("eval"), vec![1.0, 2.0]);
    }

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
        check_unobserved_error().expect("no bridge op failed");
    }
}
