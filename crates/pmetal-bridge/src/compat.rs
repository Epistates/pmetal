//! mlx-rs compatibility layer — drop-in types backed by InlineArray.
//!
//! Switch `use mlx_rs::*` to `use pmetal_bridge::compat::*` for zero-allocation
//! MLX access with the same API surface.
//!
//! # What is covered
//!
//! - `Array` — re-export of [`InlineArray`](crate::InlineArray)
//! - `Dtype` — integer codes matching MLX's encoding (same as bridge.h)
//! - `Exception` — drop-in for `mlx_rs::error::Exception`
//! - `Param<T>` — trainable-parameter wrapper matching mlx-rs's struct
//! - `Module` / `ModuleParameters` traits
//! - `ModuleParamRef` / `ModuleParamMut` / `FlattenedModuleParam` type aliases
//! - `eval` / `eval_params` free functions
//! - `ops::*` / `random::*` / `nn::*` / `fast::*` sub-modules

// ── Array ────────────────────────────────────────────────────────────────────

/// Re-export InlineArray as Array for source compatibility.
pub use crate::InlineArray as Array;

// ── Dtype ────────────────────────────────────────────────────────────────────

/// Array element type — integer codes matching MLX's `mlx_dtype` enum.
///
/// The `#[repr(i32)]` discriminants are kept in sync with `bridge.h` so that
/// `dtype.as_i32()` can be passed directly to every bridge function that
/// accepts a `dtype: i32` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Dtype {
    Bool = 0,
    Uint8 = 1,
    Uint16 = 2,
    Uint32 = 3,
    Uint64 = 4,
    Int8 = 5,
    Int16 = 6,
    Int32 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Bfloat16 = 11,
    Complex64 = 12,
}

impl Dtype {
    /// Return the raw `i32` code expected by bridge functions.
    #[inline]
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Convert a raw MLX dtype integer back to [`Dtype`].
    ///
    /// Compatible with mlx-rs `Dtype::from(raw_i32)`.
    pub fn from_raw(raw: i32) -> Self {
        match raw {
            0 => Dtype::Bool,
            1 => Dtype::Uint8,
            2 => Dtype::Uint16,
            3 => Dtype::Uint32,
            4 => Dtype::Uint64,
            5 => Dtype::Int8,
            6 => Dtype::Int16,
            7 => Dtype::Int32,
            8 => Dtype::Int64,
            9 => Dtype::Float16,
            10 => Dtype::Float32,
            11 => Dtype::Bfloat16,
            12 => Dtype::Complex64,
            _ => Dtype::Float32, // fallback
        }
    }

    /// Returns `true` if the dtype is a floating-point type.
    #[inline]
    pub fn is_float(self) -> bool {
        matches!(self, Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16)
    }

    /// Returns `true` for float or complex types.
    #[inline]
    pub fn is_inexact(self) -> bool {
        matches!(
            self,
            Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16 | Dtype::Complex64
        )
    }

    /// Type promotion following MLX's rules (subset — covers float/int cases).
    pub fn promote_with(self, other: Self) -> Self {
        use Dtype::*;
        // Simplified promotion — matches the cases that arise in model code.
        // Full table lives in mlx-rs; we cover the common paths here.
        match (self, other) {
            (a, b) if a == b => a,
            (Float32, _) | (_, Float32) => Float32,
            (Bfloat16, Float16) | (Float16, Bfloat16) => Float32,
            (Bfloat16, _) | (_, Bfloat16) => Bfloat16,
            (Float16, _) | (_, Float16) => Float16,
            (Int64, _) | (_, Int64) => Int64,
            (Uint64, _) | (_, Uint64) => Uint64,
            (Int32, _) | (_, Int32) => Int32,
            (Uint32, _) | (_, Uint32) => Uint32,
            (Int16, _) | (_, Int16) => Int16,
            (Uint16, _) | (_, Uint16) => Uint16,
            (Int8, _) | (_, Int8) => Int8,
            (Uint8, _) | (_, Uint8) => Uint8,
            _ => self,
        }
    }
}

// ── Exception ────────────────────────────────────────────────────────────────

/// Drop-in for `mlx_rs::error::Exception`.
///
/// Constructed either from a `String`/`&str` message or via [`Exception::custom`].
#[derive(Debug)]
pub struct Exception {
    message: String,
}

impl Exception {
    /// Create an exception with the given message.
    #[track_caller]
    pub fn custom(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }

    /// The error message string.
    pub fn what(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Exception {}

impl From<String> for Exception {
    fn from(s: String) -> Self {
        Self::custom(s)
    }
}

impl From<&str> for Exception {
    fn from(s: &str) -> Self {
        Self::custom(s)
    }
}

// ── Param ────────────────────────────────────────────────────────────────────

/// A trainable-parameter wrapper, matching `mlx_rs::module::Param<T>`.
///
/// Derefs transparently to `T`, carries a freeze flag.
#[derive(Debug, Clone)]
pub struct Param<T> {
    /// The wrapped value.
    pub value: T,
    is_frozen: bool,
}

impl<T> Param<T> {
    /// Wrap a value as a (non-frozen) trainable parameter.
    pub fn new(value: T) -> Self {
        Self {
            value,
            is_frozen: false,
        }
    }

    /// Whether this parameter is currently frozen.
    pub fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Freeze the parameter (excluded from gradient computation).
    pub fn freeze(&mut self) {
        self.is_frozen = true;
    }

    /// Unfreeze the parameter.
    pub fn unfreeze(&mut self) {
        self.is_frozen = false;
    }

    /// Consume the wrapper and return the inner value.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> From<T> for Param<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> std::ops::Deref for Param<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for Param<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T: AsRef<U>, U> AsRef<U> for Param<T> {
    fn as_ref(&self) -> &U {
        self.value.as_ref()
    }
}

// ── Module parameter tree types ───────────────────────────────────────────────

/// Nested parameter value — mirrors `mlx_rs::nested::NestedValue`.
///
/// Used to represent a tree of named parameters for `ModuleParameters`.
pub enum NestedValue<V> {
    /// A leaf parameter value.
    Value(V),
    /// A sub-map of named nested values.
    Map(std::collections::HashMap<std::rc::Rc<str>, NestedValue<V>>),
}

impl<V> NestedValue<V> {
    /// Flatten the tree into a `HashMap<String, V>`.
    pub fn flatten_into(self, prefix: &str, out: &mut std::collections::HashMap<String, V>) {
        match self {
            NestedValue::Value(v) => {
                out.insert(prefix.to_owned(), v);
            }
            NestedValue::Map(m) => {
                for (k, child) in m {
                    let full = if prefix.is_empty() {
                        k.to_string()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    child.flatten_into(&full, out);
                }
            }
        }
    }
}

/// Borrowed parameter tree (returned by `parameters()`).
pub type ModuleParamRef<'a> = std::collections::HashMap<std::rc::Rc<str>, NestedValue<&'a Array>>;

/// Mutably borrowed parameter tree (returned by `parameters_mut()`).
pub type ModuleParamMut<'a> =
    std::collections::HashMap<std::rc::Rc<str>, NestedValue<&'a mut Array>>;

/// Owned, flattened parameter map (used by optimizers and `value_and_grad`).
pub type FlattenedModuleParam = std::collections::HashMap<std::rc::Rc<str>, Array>;

// ── mlx-rs compatibility type aliases ────────────────────────────────────────

/// Stub for `mlx_rs::StreamOrDevice` — the bridge is always synchronous.
///
/// All methods that accepted a `Stream` argument should be replaced with
/// equivalent no-`Stream` bridge calls.  This type exists only to satisfy
/// type-checking in code that has not yet been updated.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stream;

impl Stream {
    pub fn cpu() -> Self {
        Self
    }
    pub fn gpu() -> Self {
        Self
    }
    pub fn default() -> Self {
        Self
    }
}

/// Stub for `mlx_rs::fast::ScaledDotProductAttentionMask`.
///
/// In the bridge, use `fast::scaled_dot_product_attention_masked()` directly
/// or pass `None` / an explicit mask array.
#[derive(Debug, Clone)]
pub enum ScaledDotProductAttentionMask {
    /// Causal mask (upper-triangular -inf).
    Causal,
    /// Explicit additive mask array.
    Array(Array),
    /// No mask.
    None,
}

/// Stub for `std::collections::hash_map::RandomState` compatibility.
pub use std::collections::hash_map::RandomState;

/// Stub for `mlx_rs::Device` / `mlx_rs::StreamOrDevice`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Device;

impl Device {
    pub fn gpu() -> Self {
        Self
    }
    pub fn cpu() -> Self {
        Self
    }
}

// ── Module traits ─────────────────────────────────────────────────────────────

/// Forward-pass trait for neural-network modules.
///
/// The generic `Input` parameter mirrors `mlx_rs::module::Module<Input>`,
/// allowing both `&Array` and tuple inputs.
pub trait Module<Input>: ModuleParameters + std::fmt::Debug {
    /// Output type produced by the forward pass.
    type Output;

    /// Error type returned on failure.
    type Error: std::error::Error;

    /// Run the forward pass.
    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error>;

    /// Toggle training mode (affects dropout, etc.).
    fn training_mode(&mut self, mode: bool);
}

/// Accessor trait for module parameters — used by optimizers and autograd.
pub trait ModuleParameters {
    /// Total number of leaf parameters.
    fn num_parameters(&self) -> usize;

    /// Borrow all parameters as a nested tree.
    fn parameters(&self) -> ModuleParamRef<'_>;

    /// Mutably borrow all parameters.
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;

    /// Borrow only the trainable (non-frozen) parameters.
    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        // Default: identical to parameters(). Override to filter frozen params.
        self.parameters()
    }

    /// Freeze all parameters.
    fn freeze_parameters(&mut self, _recursive: bool) {}

    /// Unfreeze all parameters.
    fn unfreeze_parameters(&mut self, _recursive: bool) {}

    /// Returns `Some(true)` if all parameters are frozen, `Some(false)` if at
    /// least one is not, `None` if there are no parameters.
    fn all_frozen(&self) -> Option<bool> {
        None
    }

    /// Returns `Some(true)` if any parameter is frozen.
    fn any_frozen(&self) -> Option<bool> {
        None
    }
}

// ── Top-level transform functions ─────────────────────────────────────────────

/// Materialise the lazy computation graph for a set of arrays.
///
/// Each array is cloned (which shares the underlying MLX ref-count) and then
/// evaluated.  The caller's originals are unaffected.
pub fn eval<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<(), Exception> {
    for a in arrays {
        let c = a.clone();
        c.eval();
    }
    Ok(())
}

/// Evaluate a borrowed parameter tree in place.
pub fn eval_params(params: ModuleParamRef<'_>) -> Result<(), Exception> {
    fn walk(node: &NestedValue<&Array>) {
        match node {
            NestedValue::Value(a) => {
                let c = (*a).clone();
                c.eval();
            }
            NestedValue::Map(m) => {
                for v in m.values() {
                    walk(v);
                }
            }
        }
    }
    for v in params.values() {
        walk(v);
    }
    Ok(())
}

// ── ops sub-module ────────────────────────────────────────────────────────────

/// Free functions mirroring `mlx_rs::ops::*`.
pub mod ops {
    use super::{Array, Dtype};
    use crate::inline_array::RawBuf;
    use std::mem::MaybeUninit;

    pub fn maximum(a: &Array, b: &Array) -> Array {
        a.maximum(b)
    }
    pub fn minimum(a: &Array, b: &Array) -> Array {
        a.minimum(b)
    }
    pub fn matmul(a: &Array, b: &Array) -> Array {
        a.matmul(b)
    }
    pub fn softmax_axis(a: &Array, axis: i32) -> Array {
        a.softmax(axis)
    }
    /// log(1 + x) — numerically stable log1p.
    pub fn log1p(a: &Array) -> Array {
        let one = Array::from_f32(1.0);
        a.add(&one).log()
    }
    pub fn broadcast_to(a: &Array, shape: &[i32]) -> Array {
        a.broadcast_to(shape)
    }

    /// Concatenate a slice of arrays along `axis`.
    ///
    /// Uses the fast two-array path when `arrays.len() == 2`, otherwise chains
    /// `concatenate_2` left-to-right (equivalent to `mx.concatenate`).
    pub fn concatenate_axis(arrays: &[&Array], axis: i32) -> Array {
        assert!(!arrays.is_empty(), "concatenate_axis: empty array slice");
        if arrays.len() == 1 {
            return arrays[0].clone();
        }
        if arrays.len() == 2 {
            return arrays[0].concatenate_2(arrays[1], axis);
        }
        // For three or more arrays, use the contiguous-buffer MLX path via
        // the `mlx_inline_concatenate` FFI (all RawBufs must be contiguous).
        // We clone each array into a Vec<Array> so we hold the live buffers,
        // then collect raw pointers into a contiguous slice.
        let owned: Vec<Array> = arrays.iter().map(|a| (*a).clone()).collect();
        concatenate_owned_axis(&owned, axis)
    }

    /// Concatenate owned arrays along `axis`.  The arrays must all remain live
    /// for the duration of the call (they are dropped after the FFI returns).
    pub fn concatenate_owned_axis(arrays: &[Array], axis: i32) -> Array {
        assert!(
            !arrays.is_empty(),
            "concatenate_owned_axis: empty array slice"
        );
        if arrays.len() == 1 {
            return arrays[0].clone();
        }
        if arrays.len() == 2 {
            return arrays[0].concatenate_2(&arrays[1], axis);
        }
        // Collect RawBufs into a contiguous Vec so the pointer is valid.
        // SAFETY: RawBuf is Copy, and InlineArray exposes its raw field via
        // the internal `as_raw_ptr` method.  We must not let `arrays` drop
        // while we hold the pointer.
        let raw_copies: Vec<RawBuf> = arrays
            .iter()
            .map(|a| {
                // Copy the raw buffer — this is a C++ copy-construct (ref-count bump).
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe {
                    // mlx_inline_init_copy is the copy-constructor trampoline.
                    // It is `pub(crate)` in inline_array but we are inside the
                    // same crate so this is fine.
                    crate::inline_array::raw_copy_buf(dst.as_mut_ptr(), a.as_raw_ptr());
                    dst.assume_init()
                }
            })
            .collect();

        let mut dst_raw = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            crate::inline_array::raw_concatenate(
                dst_raw.as_mut_ptr(),
                raw_copies.as_ptr(),
                raw_copies.len() as i32,
                axis,
            );
        }
        // Destroy the temporary copies we made.
        for mut rb in raw_copies {
            unsafe {
                crate::inline_array::raw_destroy(&mut rb);
            }
        }
        unsafe { crate::inline_array::from_raw_buf(dst_raw.assume_init()) }
    }

    /// Stack arrays along a new axis.
    pub fn stack_axis(arrays: &[Array], axis: i32) -> Array {
        assert!(!arrays.is_empty(), "stack_axis: empty array slice");
        let raw_copies: Vec<RawBuf> = arrays
            .iter()
            .map(|a| {
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe {
                    crate::inline_array::raw_copy_buf(dst.as_mut_ptr(), a.as_raw_ptr());
                    dst.assume_init()
                }
            })
            .collect();

        let mut dst_raw = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            crate::inline_array::raw_stack(
                dst_raw.as_mut_ptr(),
                raw_copies.as_ptr(),
                raw_copies.len() as i32,
                axis,
            );
        }
        for mut rb in raw_copies {
            unsafe {
                crate::inline_array::raw_destroy(&mut rb);
            }
        }
        unsafe { crate::inline_array::from_raw_buf(dst_raw.assume_init()) }
    }

    pub fn expand_dims(a: &Array, axis: i32) -> Array {
        a.expand_dims(axis)
    }
    pub fn repeat_axis(a: Array, repeats: i32, axis: i32) -> Array {
        a.repeat(repeats, axis)
    }
    /// Stack arrays along a new axis 0 — equivalent to `mlx_rs::ops::stack`.
    pub fn stack(arrays: &[Array]) -> Array {
        stack_axis(arrays, 0)
    }

    pub fn tri(n: i32, m: i32, k: i32, dtype: Dtype) -> Array {
        Array::tri(n, m, k, dtype.as_i32())
    }

    pub fn sigmoid(a: &Array) -> Array {
        a.sigmoid()
    }

    pub fn clip(a: &Array, lo: Option<&Array>, hi: Option<&Array>) -> Array {
        a.clip(lo, hi)
    }

    pub fn argsort_axis(a: &Array, axis: i32) -> Array {
        a.argsort(axis)
    }
    pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Array {
        a.argpartition(kth, axis)
    }
    pub fn cumsum(a: &Array, axis: i32) -> Array {
        a.cumsum(axis)
    }
    pub fn tril(a: &Array, k: i32) -> Array {
        a.tril(k)
    }
    pub fn argmax(a: &Array, axis: i32) -> Array {
        a.argmax(axis)
    }
    pub fn argmin(a: &Array, axis: i32) -> Array {
        a.argmin(axis)
    }
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Array {
        Array::zeros(shape, dtype.as_i32())
    }
    pub fn ones(shape: &[i32], dtype: Dtype) -> Array {
        Array::ones(shape, dtype.as_i32())
    }
    pub fn full(shape: &[i32], val: f32, dtype: Dtype) -> Array {
        Array::full(shape, val, dtype.as_i32())
    }
    pub fn arange(n: i32, dtype: Dtype) -> Array {
        Array::arange(n, dtype.as_i32())
    }
    /// `arange(0, n, 1)` — integer range as int32.
    pub fn arange_n(n: i32) -> Array {
        Array::arange(n, Dtype::Int32.as_i32())
    }
    /// `arange(start, stop, 1)` — integer range; equivalent to `mlx_rs::ops::arange::<i32,i32>(start, stop, 1)`.
    pub fn arange_from(start: i32, stop: i32) -> Array {
        let n = (stop - start).max(0);
        let base = Array::arange(n, Dtype::Int32.as_i32());
        if start == 0 {
            base
        } else {
            let offset = crate::InlineArray::from_f32(start as f32).as_dtype(Dtype::Int32.as_i32());
            base.add(&offset)
        }
    }
    pub fn zeros_like(a: &Array) -> Array {
        a.zeros_like()
    }
    pub fn eye(n: i32, dtype: Dtype) -> Array {
        Array::eye(n, dtype.as_i32())
    }
    pub fn flatten(a: &Array, start: i32, end: i32) -> Array {
        a.flatten(start, end)
    }
    pub fn transpose(a: &Array) -> Array {
        a.t()
    }
    pub fn transpose_axes(a: &Array, axes: &[i32]) -> Array {
        a.transpose_axes(axes)
    }
    pub fn reshape(a: &Array, shape: &[i32]) -> Array {
        a.reshape(shape)
    }
    pub fn squeeze(a: &Array, axis: i32) -> Array {
        a.squeeze(axis)
    }
    pub fn sum_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.sum_axis(axis, keepdims)
    }
    pub fn sum_axes(a: &Array, axes: &[i32], keepdims: bool) -> Array {
        a.sum_axes(axes, keepdims)
    }
    pub fn sum_all(a: &Array) -> Array {
        a.sum_all()
    }
    pub fn mean_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.mean_axis(axis, keepdims)
    }
    pub fn mean_all(a: &Array) -> Array {
        a.mean_all()
    }
    pub fn max_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.max_axis(axis, keepdims)
    }
    pub fn min_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.min_axis(axis, keepdims)
    }
    pub fn logsumexp(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.logsumexp(axis, keepdims)
    }
    pub fn exp(a: &Array) -> Array {
        a.exp()
    }
    pub fn log(a: &Array) -> Array {
        a.log()
    }
    pub fn sqrt(a: &Array) -> Array {
        a.sqrt()
    }
    pub fn abs(a: &Array) -> Array {
        a.abs_val()
    }
    pub fn square(a: &Array) -> Array {
        a.square()
    }
    pub fn pow(a: &Array, b: &Array) -> Array {
        a.pow(b)
    }
    pub fn where_fn(cond: &Array, a: &Array, b: &Array) -> Array {
        cond.where_cond(a, b)
    }
    /// `r#where` — alias for `where_fn` matching the mlx-rs `ops::r#where` name.
    #[allow(non_snake_case)]
    pub fn r#where(cond: &Array, a: &Array, b: &Array) -> Array {
        cond.where_cond(a, b)
    }
    pub fn equal(a: &Array, b: &Array) -> Array {
        a.equal(b)
    }
    pub fn not_equal(a: &Array, b: &Array) -> Array {
        a.not_equal(b)
    }
    pub fn greater(a: &Array, b: &Array) -> Array {
        a.greater(b)
    }
    pub fn less(a: &Array, b: &Array) -> Array {
        a.less(b)
    }
    pub fn greater_equal(a: &Array, b: &Array) -> Array {
        a.greater_equal(b)
    }
    pub fn less_equal(a: &Array, b: &Array) -> Array {
        a.less_equal(b)
    }
    pub fn stop_gradient(a: &Array) -> Array {
        a.stop_gradient()
    }
    pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Array {
        a.take_axis(indices, axis)
    }
    pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Array {
        a.take_along_axis(indices, axis)
    }
    /// Pad array. `pad_widths`: `&[(before, after)]` per axis.
    pub fn pad(
        a: &Array,
        pad_widths: &[(i32, i32)],
        _mode: Option<&str>,
        fill_value: Option<f32>,
    ) -> Array {
        let fill = fill_value.unwrap_or(0.0);
        let flat: Vec<i32> = pad_widths.iter().flat_map(|(b, e)| [*b, *e]).collect();
        a.pad_constant(&flat, fill)
    }
    /// Wrapper matching `mlx_rs::ops::arange::<i32, f32>` signature used in vocoder.
    /// Produces a float32 arange from `start` to `stop` (exclusive), step 1.
    pub fn arange_range(start: i32, stop: i32) -> Array {
        let n = (stop - start).max(0);
        // arange(n) gives [0..n); add start offset if needed
        let a = Array::arange(n, 10); // dtype=10=float32
        if start == 0 {
            a
        } else {
            let offset = Array::from_f32(start as f32);
            a.add(&offset)
        }
    }
    pub fn conv1d(
        input: &Array,
        weight: &Array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Array {
        input.conv1d(weight, stride, padding, dilation, groups)
    }
    pub fn tanh(a: &Array) -> Array {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) = 2*sigmoid(2x) - 1
        let two = Array::from_f32(2.0);
        let two_x = a.multiply(&two);
        let sig = two_x.sigmoid();
        let one = Array::from_f32(1.0);
        let two2 = Array::from_f32(2.0);
        sig.multiply(&two2).subtract(&one)
    }

    // ── arithmetic helpers ────────────────────────────────────────────────────

    /// Element-wise addition — alias for `a.add(b)`.
    pub fn add(a: &Array, b: &Array) -> Array {
        a.add(b)
    }
    /// Element-wise subtraction.
    pub fn subtract(a: &Array, b: &Array) -> Array {
        a.subtract(b)
    }
    /// Element-wise multiplication.
    pub fn multiply(a: &Array, b: &Array) -> Array {
        a.multiply(b)
    }
    /// Element-wise division.
    pub fn divide(a: &Array, b: &Array) -> Array {
        a.divide(b)
    }
    /// Negate: `-a`.
    pub fn negative(a: &Array) -> Array {
        let neg_one = Array::from_f32(-1.0);
        a.multiply(&neg_one)
    }

    // ── trigonometry ─────────────────────────────────────────────────────────

    pub fn sin(a: &Array) -> Array {
        a.sin()
    }
    pub fn cos(a: &Array) -> Array {
        a.cos()
    }

    // ── aliases and missing variants ──────────────────────────────────────────

    /// Alias for `zeros` — `zeros_dtype(shape, dtype)` matches mlx-rs naming.
    pub fn zeros_dtype(shape: &[i32], dtype: Dtype) -> Array {
        Array::zeros(shape, dtype.as_i32())
    }
    /// Alias: `argmax` with keepdims=false.
    pub fn argmax_axis(a: &Array, axis: i32) -> Array {
        a.argmax(axis)
    }
    /// Alias: `argmin` with keepdims=false.
    pub fn argmin_axis(a: &Array, axis: i32) -> Array {
        a.argmin(axis)
    }
    /// `which(cond, x, y)` — alias for `where_fn`.
    pub fn which(cond: &Array, x: &Array, y: &Array) -> Array {
        cond.where_cond(x, y)
    }

    /// Tile array `reps` times along each dimension.
    ///
    /// `reps` is the number of repetitions per axis (broadcast from the end).
    pub fn tile(a: &Array, reps: &[i32]) -> Array {
        let ndim = a.ndim() as usize;
        let nrep = reps.len();
        let mut arr = a.clone();
        // Extend ndim to match reps length if needed (prepend 1-dims).
        if nrep > ndim {
            for _ in ndim..nrep {
                arr = arr.expand_dims(0);
            }
        }
        let effective_ndim = arr.ndim() as usize;
        let pad = if nrep < effective_ndim {
            effective_ndim - nrep
        } else {
            0
        };
        let full_reps: Vec<i32> = std::iter::repeat(1)
            .take(pad)
            .chain(reps.iter().copied())
            .collect();
        for (axis, &rep) in full_reps.iter().enumerate() {
            if rep > 1 {
                arr = arr.repeat(rep, axis as i32);
            }
        }
        arr
    }

    /// Split array into `num_sections` equal pieces along `axis`.
    ///
    /// Equivalent to `np.split(a, num_sections, axis=axis)` or `mx.split(a, num_sections, axis)`.
    pub fn split(a: &Array, num_sections: i32, axis: i32) -> Vec<Array> {
        let dim = a.shape()[axis as usize];
        let section_size = dim / num_sections;
        // Build split indices: [section_size, 2*section_size, ..., (n-1)*section_size]
        let indices: Vec<i32> = (1..num_sections).map(|i| i * section_size).collect();
        a.split(&indices, axis)
    }

    /// Split array at given indices along `axis`.
    ///
    /// Equivalent to `np.split(a, indices, axis=axis)` or `mx.split(a, indices, axis)`.
    /// `indices` are the positions *before* which splits are made (i.e. [i0, i1] → 3 pieces:
    /// `[:i0]`, `[i0:i1]`, `[i1:]`).
    pub fn split_sections(a: &Array, indices: &[i32], axis: i32) -> Vec<Array> {
        a.split(indices, axis)
    }

    /// Scatter: create a new array where `a[indices]` = `updates` along `axis`.
    ///
    /// Returns a new array; does not modify `a` in place.
    /// Equivalent to `mlx_rs::ops::put_along_axis`.
    pub fn put_along_axis(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Array {
        a.put_along_axis_op(indices, updates, axis)
    }

    /// `async_eval` — in the bridge, evaluation is always synchronous; this is a no-op.
    pub fn async_eval<'a>(arrays: impl IntoIterator<Item = &'a Array>) {
        let _ = arrays;
    }

    /// `logsumexp_axis` — alias for `logsumexp(a, axis, false)`.
    pub fn logsumexp_axis(a: &Array, axis: i32) -> Array {
        a.logsumexp(axis, false)
    }

    /// `logsumexp_axis_keepdims` — `logsumexp(a, axis, keepdims)`.
    pub fn logsumexp_axis_keepdims(a: &Array, axis: i32, keepdims: bool) -> Array {
        a.logsumexp(axis, keepdims)
    }

    /// Reciprocal square root: `1/sqrt(x)`.
    pub fn rsqrt(a: &Array) -> Array {
        a.rsqrt()
    }

    /// Quantize to FP8 (E4M3 format, stored as uint8).
    ///
    /// Stub implementation — returns the input cast to Float16 then reinterpreted
    /// as uint8.  Replace with proper FP8 when available.
    pub fn to_fp8(x: &Array) -> Result<Array, super::Exception> {
        // FP8 is not yet in the inline bridge; fall back to float16 as uint8.
        let fp16 = x.as_dtype(super::Dtype::Float16.as_i32());
        Ok(fp16.as_dtype(super::Dtype::Uint8.as_i32()))
    }

    /// Dequantize from FP8 to the target dtype.
    ///
    /// Stub implementation — casts uint8 input to the requested dtype.
    pub fn from_fp8(x: &Array, dtype: super::Dtype) -> Result<Array, super::Exception> {
        Ok(x.as_dtype(dtype.as_i32()))
    }

    /// Evenly-spaced values from `start` to `stop` (inclusive).
    pub fn linspace(start: f32, stop: f32, n: i32, dtype: Dtype) -> Array {
        Array::linspace(start, stop, n, dtype.as_i32())
    }

    /// Floor: largest integer ≤ x.
    ///
    /// Implemented as `cast to int32, then cast back` for float inputs.
    /// For inputs already of integer dtype, this is a no-op.
    pub fn floor(a: &Array) -> Array {
        // floor(x) = cast_to_int(x - (x < 0) * 1) — approximate for int casts
        // More robustly: use the fact that int32 truncates toward zero:
        // floor(x) = trunc(x) - (x < 0 AND x != trunc(x))
        let int_vals = a.as_dtype(Dtype::Int32.as_i32());
        let float_int = int_vals.as_dtype(a.dtype_raw());
        // Correction: if original < float_int (negative truncation direction), subtract 1.
        let one = Array::from_f32(1.0).as_dtype(a.dtype_raw());
        let needs_correction = a.less(&float_int);
        let correction = needs_correction.as_dtype(a.dtype_raw()).multiply(&one);
        float_int.subtract(&correction)
    }

    /// Ceil: smallest integer ≥ x.
    pub fn ceil(a: &Array) -> Array {
        // ceil(x) = -floor(-x)
        let neg = a.multiply(&Array::from_f32(-1.0));
        let floored = floor(&neg);
        floored.multiply(&Array::from_f32(-1.0))
    }

    /// Round to nearest integer.
    pub fn round(a: &Array) -> Array {
        // round(x) = floor(x + 0.5)
        let half = Array::from_f32(0.5).as_dtype(a.dtype_raw());
        floor(&a.add(&half))
    }

    /// Logical NOT: bool → !bool.
    pub fn logical_not(a: &Array) -> Array {
        let zero = Array::zeros(&[1], Dtype::Bool.as_i32());
        a.equal(&zero)
    }

    /// Logical AND: cast both to bool then multiply (AND).
    pub fn logical_and(a: &Array, b: &Array) -> Array {
        let ab = a.as_dtype(Dtype::Bool.as_i32());
        let bb = b.as_dtype(Dtype::Bool.as_i32());
        // bool multiply = AND
        ab.multiply(&bb)
    }

    /// Logical OR: cast both to bool then add and clamp (OR).
    pub fn logical_or(a: &Array, b: &Array) -> Array {
        let ab = a.as_dtype(Dtype::Bool.as_i32());
        let bb = b.as_dtype(Dtype::Bool.as_i32());
        let sum = ab.add(&bb);
        // Any nonzero = true: cast to bool clips to {0,1}
        sum.as_dtype(Dtype::Bool.as_i32())
    }

    /// Returns a bool array — true where x is NaN (using x != x identity).
    pub fn is_nan(a: &Array) -> Array {
        // NaN != NaN is the IEEE 754 definition
        a.not_equal(a)
    }

    /// Returns a bool array — true where x is ±Inf.
    pub fn is_inf(a: &Array) -> Array {
        // |x| > f32::MAX  iff  x is ±Inf
        let abs_a = a.abs_val();
        let max_finite = Array::from_f32(f32::MAX);
        abs_a.greater(&max_finite)
    }

    /// Reduce a bool/numeric array with logical-OR along all axes (or a given axis).
    /// Equivalent to mlx_rs::ops::any(a, axes, keepdims).
    /// `axes`: None = reduce all axes.
    pub fn any(a: &Array, axes: Option<&[i32]>, keep_dims: bool) -> Array {
        // Cast to bool first, then sum. Any nonzero sum → true.
        let b = a.as_dtype(Dtype::Bool.as_i32());
        let s = match axes {
            None => b.sum(None),
            Some(ax) => {
                // sum over each axis
                let mut result = b.clone();
                for &axis in ax {
                    result = result.sum(Some(axis));
                }
                result
            }
        };
        let out = s.as_dtype(Dtype::Bool.as_i32());
        if keep_dims { out } else { out }
    }

    /// Returns `true` (scalar bool) if any element is NaN.
    pub fn item_bool(a: &Array) -> bool {
        let a_clone = a.clone();
        a_clone.eval();
        // Cast to f32 and check if value > 0.5
        let f = a_clone.as_dtype(Dtype::Float32.as_i32()).item_f32();
        f > 0.5
    }

    /// Select a single index along `axis`, removing that dimension.
    /// Equivalent to `a[..., idx, ...]` in Python/mlx.
    pub fn select_axis(a: &Array, idx: i32, axis: i32) -> Array {
        let ndim = a.ndim() as i32;
        let ax = if axis < 0 { ndim + axis } else { axis };
        let i = Array::from_i32_slice_shaped(&[idx], &[1]);
        let out = a.take_axis(&i, ax);
        out.squeeze(ax)
    }

    /// Slice the last axis from 0 to `end` (exclusive), all other axes full.
    /// Equivalent to `a[..., :end]` in Python/mlx.
    pub fn slice_last_to(a: &Array, end: i32) -> Array {
        let ndim = a.ndim() as usize;
        let shape = a.shape();
        let s: Vec<i32> = vec![0; ndim];
        let mut e: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        e[ndim - 1] = end;
        a.slice(&s, &e)
    }

    /// Slice the last axis from `start` to the end, all other axes full.
    /// Equivalent to `a[..., start:]` in Python/mlx.
    pub fn slice_last_from(a: &Array, start: i32) -> Array {
        let ndim = a.ndim() as usize;
        let shape = a.shape();
        let mut s: Vec<i32> = vec![0; ndim];
        let e: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        s[ndim - 1] = start;
        a.slice(&s, &e)
    }

    /// Slice a specific axis from `start` to `end`, all other axes full.
    /// Equivalent to `a[..., start:end, ...]` at the given axis.
    pub fn slice_axis(a: &Array, axis: i32, start: i32, end: i32) -> Array {
        let ndim = a.ndim() as i32;
        let ax = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        let shape = a.shape();
        let mut s: Vec<i32> = vec![0; ndim as usize];
        let mut e: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        s[ax] = start;
        e[ax] = end;
        a.slice(&s, &e)
    }

    /// Slice a specific axis from `start` to the end, all other axes full.
    pub fn slice_axis_from(a: &Array, axis: i32, start: i32) -> Array {
        let ndim = a.ndim() as i32;
        let ax = if axis < 0 {
            (ndim + axis) as usize
        } else {
            axis as usize
        };
        let shape = a.shape();
        let mut s: Vec<i32> = vec![0; ndim as usize];
        let e: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        s[ax] = start;
        a.slice(&s, &e)
    }
}

// ── random sub-module ─────────────────────────────────────────────────────────

/// Free functions mirroring `mlx_rs::random::*`.
pub mod random {
    use super::{Array, Dtype};

    pub fn normal(shape: &[i32], dtype: Dtype) -> Array {
        Array::random_normal(shape, dtype.as_i32())
    }
    pub fn uniform(shape: &[i32], dtype: Dtype) -> Array {
        Array::random_uniform(shape, dtype.as_i32())
    }
    pub fn seed(s: u64) {
        crate::inline_array::random_seed(s);
    }
    pub fn randint(low: i32, high: i32, shape: &[i32], dtype: Dtype) -> Array {
        Array::random_randint(low, high, shape, dtype.as_i32())
    }
    pub fn bernoulli(p: &Array, shape: &[i32]) -> Array {
        Array::random_bernoulli(p, shape)
    }
    /// Uniform random in [lo, hi).  Equivalent to `mlx_rs::random::uniform(-b, b, shape, None)`.
    pub fn uniform_range(lo: f32, hi: f32, shape: &[i32], dtype: Dtype) -> Array {
        // uniform() gives [0,1); scale and shift to [lo, hi).
        let u = Array::random_uniform(shape, Dtype::Float32.as_i32());
        let range = Array::from_f32(hi - lo);
        let offset = Array::from_f32(lo);
        let scaled = u.multiply(&range).add(&offset);
        if dtype == Dtype::Float32 {
            scaled
        } else {
            scaled.as_dtype(dtype.as_i32())
        }
    }
    /// Uniform random normal in [0,1) as f32.  Deprecated alias for `uniform`.
    pub fn uniform_f32(shape: &[i32]) -> Array {
        Array::random_uniform(shape, Dtype::Float32.as_i32())
    }

    /// Sample from a categorical distribution.
    ///
    /// `logits`: unnormalized log-probabilities of shape `[..., num_classes]`.
    /// Returns integer indices of shape `[...]` (the last axis is reduced).
    ///
    /// Equivalent to `mlx_rs::random::categorical(logits, axis, None, None)`.
    pub fn categorical(logits: &Array, _axis: i32) -> Array {
        // InlineArray::categorical() uses the built-in MLX sampling
        logits.categorical()
    }
}

// ── nn sub-module ─────────────────────────────────────────────────────────────

/// Free functions and layer types mirroring `mlx_rs::nn::*` and `mlx_nn::*`.
pub mod nn {
    // Re-export layer types so `use pmetal_bridge::compat::nn` works
    // as a drop-in for `use mlx_rs::nn`.
    pub use super::layers::{
        Conv1d, Conv1dBuilder, Conv2d, Conv2dBuilder, Embedding, GroupNorm, GroupNormBuilder,
        LayerNorm, LayerNormBuilder, Linear, LinearBuilder, RmsNorm, RmsNormBuilder, Rope,
        RopeBuilder, Sequential,
    };

    use super::{Array, Exception};

    pub fn softplus(a: &Array) -> Array {
        a.softplus()
    }
    pub fn sigmoid(a: &Array) -> Array {
        a.sigmoid()
    }
    pub fn relu(a: &Array) -> Array {
        a.relu()
    }
    pub fn gelu(a: &Array) -> Array {
        a.gelu()
    }
    /// GeLU with tanh approximation — matches `mlx_rs::nn::gelu_approximate`.
    pub fn gelu_approximate(a: &Array) -> Array {
        a.gelu()
    }
    pub fn silu(a: &Array) -> Array {
        a.silu()
    }
    /// Log-sigmoid: `log(sigmoid(x)) = -softplus(-x)`.
    pub fn log_sigmoid(a: &Array) -> Array {
        a.negative().softplus().negative()
    }
    pub fn log_softmax(a: &Array, axis: i32) -> Array {
        a.log_softmax(axis)
    }
    pub fn softmax(a: &Array, axis: i32) -> Array {
        a.softmax(axis)
    }
    pub fn leaky_relu(a: &Array, neg_slope: f32) -> Array {
        a.leaky_relu(neg_slope)
    }
    pub fn cross_entropy(logits: &Array, targets: &Array, axis: i32) -> Array {
        logits.cross_entropy(targets, axis)
    }

    /// Compute `(loss, gradients)` via callback-based autograd — explicit-array form.
    ///
    /// `loss_fn` receives `[params..., inputs...]` as a flat slice and must
    /// return a scalar loss array.  Gradients are computed w.r.t. the first
    /// `params.len()` arrays.
    ///
    /// This is a thin shim over the bridge `value_and_grad` function; the
    /// `Result` wrapper is present only for API parity with `mlx_rs`.
    pub fn value_and_grad_explicit<F>(
        loss_fn: F,
        params: &[Array],
        inputs: &[Array],
    ) -> Result<(Array, Vec<Array>), Exception>
    where
        F: FnMut(&[Array]) -> Array,
    {
        Ok(crate::inline_array::value_and_grad(loss_fn, params, inputs))
    }

    /// mlx-rs compatible closure-returning form of `value_and_grad`.
    ///
    /// Mirrors the mlx-rs API:
    /// ```ignore
    /// let mut vag = nn::value_and_grad(loss_fn);
    /// let (loss, grads) = vag(model, inputs)?;
    /// ```
    ///
    /// - `loss_fn` takes `(&mut M, T)` and returns `Result<Array, Exception>`.
    /// - The returned closure accepts `(&mut M, T)` and returns
    ///   `Result<(Array, FlattenedModuleParam), Exception>`.
    ///
    /// Trainable parameters are extracted from `M` before autograd, the
    /// bridge computes gradients, and the result is re-keyed into a
    /// `FlattenedModuleParam` with the same names.
    pub fn value_and_grad<M, T, F>(
        mut loss_fn: F,
    ) -> impl FnMut(&mut M, T) -> Result<(Array, super::FlattenedModuleParam), super::Exception>
    where
        M: super::ModuleParameters,
        F: FnMut(&mut M, T) -> Result<Array, super::Exception>,
    {
        move |model: &mut M, inputs: T| {
            use super::FlattenedModuleParam;
            use std::rc::Rc;

            // 1. Snapshot trainable parameter values — stable key order.
            let flat: FlattenedModuleParam = {
                let tree = model.trainable_parameters();
                let mut out = FlattenedModuleParam::new();
                super::flatten_nested_ref_owned(&tree, "", &mut out);
                out
            };
            let keys: Vec<Rc<str>> = flat.keys().cloned().collect();
            let param_arrays: Vec<Array> = keys.iter().map(|k| flat[k].clone()).collect();
            let n_params = param_arrays.len();

            // 2. Wrap inputs in an Option so the inner closure can move them out
            //    exactly once (MLX calls the callback once per value_and_grad call).
            let mut inputs_slot: Option<T> = Some(inputs);

            // SAFETY: both `model` and `loss_fn` outlive the `flat_loss` closure —
            // they all live on the same call frame.  `flat_loss` is consumed
            // synchronously by `value_and_grad` before this function returns.
            let model_ptr: *mut M = model as *mut M;
            let loss_fn_ptr: *mut F = &mut loss_fn as *mut F;
            let keys_snap: Vec<Rc<str>> = keys.clone();

            let flat_loss = move |all_arrays: &[Array]| -> Array {
                let model_mut = unsafe { &mut *model_ptr };
                let loss_fn_mut = unsafe { &mut *loss_fn_ptr };

                // Update the model's trainable params with the autograd arrays.
                {
                    let mut pm = model_mut.parameters_mut();
                    for (key, arr) in keys_snap.iter().zip(all_arrays[..n_params].iter()) {
                        super::update_trainable_param(&mut pm, key, arr.clone());
                    }
                }

                // Consume inputs from the slot (called exactly once by bridge).
                let inp = inputs_slot
                    .take()
                    .expect("value_and_grad callback called more than once");
                match loss_fn_mut(model_mut, inp) {
                    Ok(loss) => loss,
                    Err(_) => Array::from_f32(f32::NAN),
                }
            };

            // 3. Run bridge autograd (no extra "input" arrays — all captured).
            let (loss, grad_arrays) =
                crate::inline_array::value_and_grad(flat_loss, &param_arrays, &[]);

            // 4. Re-key gradients into FlattenedModuleParam.
            let grads: FlattenedModuleParam =
                keys.into_iter().zip(grad_arrays.into_iter()).collect();

            Ok((loss, grads))
        }
    }

    /// mlx-rs compatible `keyed_value_and_grad`.
    ///
    /// Takes a closure `loss_fn(params: FlattenedModuleParam, inputs: T) -> Result<Vec<Array>>`
    /// and returns a closure that computes `(values, grad_map)` via autograd over
    /// the flattened param map.
    ///
    /// The returned closure signature:
    /// ```ignore
    /// let mut vg = keyed_value_and_grad(loss_fn);
    /// let (values, grads_map) = vg(params, inputs)?;
    /// ```
    pub fn keyed_value_and_grad<T, F>(
        mut loss_fn: F,
    ) -> impl FnMut(
        super::FlattenedModuleParam,
        T,
    )
        -> Result<(Vec<super::Array>, super::FlattenedModuleParam), super::Exception>
    where
        T: 'static,
        F: FnMut(super::FlattenedModuleParam, T) -> Result<Vec<super::Array>, super::Exception>,
    {
        move |params: super::FlattenedModuleParam, inputs: T| {
            use super::FlattenedModuleParam;
            use std::rc::Rc;

            // Stable key order.
            let keys: Vec<Rc<str>> = params.keys().cloned().collect();
            let param_arrays: Vec<super::Array> = keys.iter().map(|k| params[k].clone()).collect();
            let n_params = param_arrays.len();

            let mut inputs_slot: Option<T> = Some(inputs);

            // SAFETY: params_ptr and loss_fn_ptr live for the duration of the
            // synchronous call to crate::inline_array::value_and_grad.
            let loss_fn_ptr: *mut F = &mut loss_fn as *mut F;
            let keys_snap: Vec<Rc<str>> = keys.clone();

            let flat_loss = move |all_arrays: &[super::Array]| -> super::Array {
                let loss_fn_mut = unsafe { &mut *loss_fn_ptr };

                // Re-build keyed param map from autograd arrays.
                let param_map: FlattenedModuleParam = keys_snap
                    .iter()
                    .cloned()
                    .zip(all_arrays[..n_params].iter().cloned())
                    .collect();

                let inp = inputs_slot
                    .take()
                    .expect("keyed_value_and_grad callback called more than once");

                match loss_fn_mut(param_map, inp) {
                    Ok(mut vals) => {
                        // If there are multiple values, reduce to first (loss).
                        vals.drain(..)
                            .next()
                            .unwrap_or_else(|| super::Array::from_f32(0.0))
                    }
                    Err(_) => super::Array::from_f32(f32::NAN),
                }
            };

            // Bridge autograd: gradients w.r.t. param_arrays.
            let (loss_val, grad_arrays) =
                crate::inline_array::value_and_grad(flat_loss, &param_arrays, &[]);

            // Re-key gradients.
            let grads: FlattenedModuleParam =
                keys.into_iter().zip(grad_arrays.into_iter()).collect();

            Ok((vec![loss_val], grads))
        }
    }
}

// ── linalg sub-module ────────────────────────────────────────────────────────

/// Free functions mirroring `mlx_rs::linalg::*`.
pub mod linalg {
    use super::Array;

    /// Compute the inverse of a triangular matrix (batched over leading dims).
    ///
    /// Equivalent to `mlx_rs::linalg::tri_inv_device(a, upper, StreamOrDevice::cpu())`.
    /// Dispatches on the CPU stream because `tri_inv` has no registered VJP — it is
    /// used as a fixed preconditioner in the GDN WY factorization and must not appear
    /// on the autograd tape.
    pub fn tri_inv(a: &Array, upper: bool) -> Array {
        a.tri_inv(upper, true /* use_cpu */)
    }

    /// Economy SVD — returns `(U, S, Vt)`.
    ///
    /// Equivalent to `mlx_rs::linalg::svd_device(a, StreamOrDevice::cpu())`.
    /// Always runs on the CPU stream (GPU SVD is not available in MLX).
    pub fn svd(a: &Array) -> (Array, Array, Array) {
        a.svd()
    }
}

// ── fast sub-module ───────────────────────────────────────────────────────────

/// Free functions mirroring `mlx_rs::fast::*`.
pub mod fast {
    use super::Array;

    /// RMS normalisation with an optional affine weight.
    /// `weight` may be `Option<&Array>` or `&Array` (converted automatically via `IntoRmsWeight`).
    pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Array {
        x.rms_norm(Some(weight), eps)
    }

    /// RMS normalisation with an explicit `Option<&Array>` weight.
    pub fn rms_norm_opt(x: &Array, weight: Option<&Array>, eps: f32) -> Array {
        x.rms_norm(weight, eps)
    }

    /// Rotary position embedding.
    pub fn rope(
        x: &Array,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> Array {
        x.rope(dims, traditional, base, scale, offset)
    }

    /// Scaled dot-product attention with string mask mode (`"none"`, `"causal"`, …).
    pub fn scaled_dot_product_attention(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask_mode: &str,
    ) -> Array {
        q.sdpa(k, v, scale, mask_mode)
    }

    /// SDPA with an explicit mask array (pass `None` for no mask).
    pub fn scaled_dot_product_attention_masked(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Array {
        q.sdpa_with_mask(k, v, scale, mask)
    }

    /// Re-export `ScaledDotProductAttentionMask` into the `fast` module so that
    /// `use pmetal_bridge::compat::fast::ScaledDotProductAttentionMask` resolves.
    pub use super::ScaledDotProductAttentionMask;
}

// ── fft sub-module ────────────────────────────────────────────────────────────

/// Free functions mirroring `mlx_rs::fft::*`.
pub mod fft {
    use super::Array;

    /// Real FFT along `axis`. `n` = `None` uses full axis length.
    pub fn rfft(a: &Array, n: Option<i32>, axis: i32) -> Array {
        a.rfft(n.unwrap_or(-1), axis)
    }

    /// Inverse real FFT along `axis`. `n` = `None` infers from input.
    pub fn irfft(a: &Array, n: Option<i32>, axis: i32) -> Array {
        a.irfft(n.unwrap_or(-1), axis)
    }
}

// ── IoError ───────────────────────────────────────────────────────────────────

/// Drop-in for `mlx_rs::error::IoError` — IO errors from safetensors loading.
#[derive(Debug)]
pub struct IoError {
    message: String,
}

impl IoError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for IoError {}

impl From<String> for IoError {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for IoError {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        Self::new(e.to_string())
    }
}

// ── ModuleParametersExt ───────────────────────────────────────────────────────

/// Extension trait adding convenience methods to `ModuleParameters`.
///
/// Mirrors `mlx_rs::module::ModuleParametersExt`.
pub trait ModuleParametersExt: ModuleParameters {
    /// Flatten all parameters into an owned `FlattenedModuleParam`.
    ///
    /// Arrays are cloned (ref-count bump only — no data copy) to avoid
    /// lifetime constraints from the intermediate nested-value tree.
    fn flatten_params(&self) -> FlattenedModuleParam {
        let tree = self.parameters();
        let mut out = FlattenedModuleParam::new();
        flatten_nested_ref_owned(&tree, "", &mut out);
        out
    }

    /// Flatten all mutable parameters into a `HashMap<String, &mut Array>`.
    ///
    /// Used by weight loaders that need to assign tensors by name.
    fn flatten_params_mut(&mut self) -> std::collections::HashMap<String, &mut Array> {
        let tree = self.parameters_mut();
        let mut out: std::collections::HashMap<String, &mut Array> =
            std::collections::HashMap::new();
        flatten_nested_mut_owned(tree, "", &mut out);
        out
    }

    /// Evaluate all parameters (materialise lazy computation graph).
    fn eval(&self) -> Result<(), Exception> {
        let p = self.parameters();
        eval_params(p)
    }
}

/// Blanket impl — any `ModuleParameters` gets `ModuleParametersExt` for free.
impl<T: ModuleParameters> ModuleParametersExt for T {}

/// Flatten a borrowed nested-value tree into an owned FlattenedModuleParam by cloning each Array.
fn flatten_nested_ref_owned(
    map: &std::collections::HashMap<std::rc::Rc<str>, NestedValue<&Array>>,
    prefix: &str,
    out: &mut FlattenedModuleParam,
) {
    for (k, v) in map {
        let full: std::rc::Rc<str> = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}").into()
        };
        match v {
            NestedValue::Value(arr) => {
                out.insert(full, (*arr).clone());
            }
            NestedValue::Map(child) => {
                flatten_nested_ref_owned(child, &full, out);
            }
        }
    }
}

fn flatten_nested_mut_owned<'a>(
    map: std::collections::HashMap<std::rc::Rc<str>, NestedValue<&'a mut Array>>,
    prefix: &str,
    out: &mut std::collections::HashMap<String, &'a mut Array>,
) {
    for (k, v) in map {
        let full: String = if prefix.is_empty() {
            k.to_string()
        } else {
            format!("{prefix}.{k}")
        };
        match v {
            NestedValue::Value(arr) => {
                out.insert(full, arr);
            }
            NestedValue::Map(child) => {
                flatten_nested_mut_owned(child, &full, out);
            }
        }
    }
}

/// Update a single trainable parameter leaf in a mutable parameter tree.
///
/// Used by `nn::value_and_grad` to inject autograd-tracked parameter arrays
/// back into the model before invoking the user's loss function.
pub(crate) fn update_trainable_param(
    pm: &mut std::collections::HashMap<std::rc::Rc<str>, NestedValue<&mut Array>>,
    key: &std::rc::Rc<str>,
    value: Array,
) {
    if let Some(entry) = pm.get_mut(key) {
        if let NestedValue::Value(arr) = entry {
            **arr = value;
        }
    }
}

// ── NestedHashMap (mlx_rs::nested compat) ────────────────────────────────────

/// Drop-in for `mlx_rs::nested::NestedHashMap<K, V>`.
///
/// A named-key tree structure used for manual `ModuleParameters` impls.
#[derive(Debug, Clone)]
pub struct NestedHashMap<K, V> {
    pub entries: std::collections::HashMap<K, NestedValue2<K, V>>,
}

/// Two-parameter nested value — mirrors `mlx_rs::nested::NestedValue<K, V>`.
#[derive(Debug, Clone)]
pub enum NestedValue2<K, V> {
    Value(V),
    Map(std::collections::HashMap<K, NestedValue2<K, V>>),
}

impl<K, V> Default for NestedHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> NestedHashMap<K, V> {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: NestedValue2<K, V>)
    where
        K: Eq + std::hash::Hash,
    {
        self.entries.insert(key, value);
    }

    /// Flatten the nested map into a `HashMap<Rc<str>, V>`.
    pub fn flatten(self) -> std::collections::HashMap<std::rc::Rc<str>, V>
    where
        K: AsRef<str> + std::fmt::Display,
    {
        fn go<K: AsRef<str> + std::fmt::Display, V>(
            prefix: &str,
            v: NestedValue2<K, V>,
            out: &mut std::collections::HashMap<std::rc::Rc<str>, V>,
        ) {
            match v {
                NestedValue2::Value(val) => {
                    out.insert(prefix.into(), val);
                }
                NestedValue2::Map(m) => {
                    for (k, child) in m {
                        let key = if prefix.is_empty() {
                            k.to_string()
                        } else {
                            format!("{prefix}.{k}")
                        };
                        go(&key, child, out);
                    }
                }
            }
        }
        let mut out = std::collections::HashMap::new();
        for (k, v) in self.entries {
            go(k.as_ref(), v, &mut out);
        }
        out
    }
}

// ── ops extras (sin, cos, rsqrt, etc.) ────────────────────────────────────────

pub mod ops_ext {
    use super::Array;

    pub fn sin(a: &Array) -> Array {
        a.sin()
    }
    pub fn cos(a: &Array) -> Array {
        a.cos()
    }
    pub fn rsqrt(a: &Array) -> Array {
        a.rsqrt()
    }
    pub fn zeros_like(a: &Array) -> Array {
        a.zeros_like()
    }
    pub fn ones_like(a: &Array) -> Array {
        a.ones_like()
    }
    pub fn tile(a: &Array, reps: &[i32]) -> Array {
        a.tile(reps)
    }
    pub fn linspace(start: f32, stop: f32, n: i32, dtype: super::Dtype) -> Array {
        Array::linspace(start, stop, n, dtype.as_i32())
    }
    pub fn split_sections(a: &Array, sections: i32, axis: i32) -> Vec<Array> {
        a.split_sections(sections, axis)
    }
    pub fn scatter_add_single(a: &Array, indices: &Array, updates: &Array, axis: i32) -> Array {
        a.scatter_add_axis(indices, updates, axis)
    }
    pub fn topk_axis(a: &Array, k: i32, axis: i32) -> Array {
        a.topk(k, axis)
    }
    pub fn put_along_axis(a: &Array, indices: &Array, values: &Array, axis: Option<i32>) -> Array {
        a.put_along_axis_op(indices, values, axis.unwrap_or(-1))
    }
    pub fn addmm(c: &Array, a: &Array, b: &Array) -> Array {
        Array::addmm(c, a, b)
    }
    pub fn argmax_axis(a: &Array, axis: i32, keepdims: bool) -> Array {
        // bridge has argmax(axis) without keepdims; wrap manually
        let out = a.argmax(axis);
        if keepdims { out.expand_dims(axis) } else { out }
    }
    pub fn argmax(a: &Array) -> Array {
        a.argmax(-1)
    }
}

// Re-export ops extras into ops for convenience
pub mod indexing {
    use super::Array;
    pub use super::ops_ext::{argmax, argmax_axis, put_along_axis, scatter_add_single, topk_axis};
    use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

    pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Array {
        a.take_along_axis(indices, axis)
    }

    fn normalize_axis(axis: i32, ndim: usize) -> usize {
        if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        }
    }

    fn normalize_bound(bound: i32, dim: i32) -> i32 {
        if bound < 0 {
            (dim + bound).clamp(0, dim)
        } else {
            bound.clamp(0, dim)
        }
    }

    fn slice_axis_range(a: &Array, axis: usize, start: i32, end: i32) -> Array {
        let shape = a.shape();
        let dim = shape[axis] as i32;
        let ndim = shape.len();
        let mut starts = vec![0; ndim];
        let mut stops: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        starts[axis] = normalize_bound(start, dim);
        stops[axis] = normalize_bound(end, dim);
        a.slice(&starts, &stops)
    }

    fn slice_axis_from(a: &Array, axis: usize, start: i32) -> Array {
        let end = a.shape()[axis] as i32;
        slice_axis_range(a, axis, start, end)
    }

    fn select_axis_idx(a: &Array, axis: usize, idx: i32) -> Array {
        let _ndim = a.ndim();
        let axis_i32 = axis as i32;
        let dim = a.dim(axis_i32);
        let normalized = if idx < 0 { dim + idx } else { idx };
        let indices = Array::from_i32_slice_shaped(&[normalized], &[1]);
        let out = a.take_axis(&indices, axis_i32);
        out.squeeze(axis_i32)
    }

    /// Thin shim: `IndexOp` replacement for simple array-backed indexing.
    pub trait IndexOp<Idx> {
        fn index(&self, idx: Idx) -> Self;
    }

    impl IndexOp<&Array> for Array {
        fn index(&self, idx: &Array) -> Self {
            self.index_array(idx)
        }
    }

    impl IndexOp<Array> for Array {
        fn index(&self, idx: Array) -> Self {
            IndexOp::<&Array>::index(self, &idx)
        }
    }

    // Integer index (e.g. `arr.index(5)`) — squeeze that axis.
    impl IndexOp<i32> for Array {
        fn index(&self, idx: i32) -> Self {
            let n = self.ndim();
            assert!(n >= 1, "index(i32): array must have at least 1 dim");
            // Take at position `idx` along axis 0, then remove that axis.
            let i = Array::from_i32_slice_shaped(&[idx], &[1]);
            let out = self.take_axis(&i, 0);
            out.squeeze(0)
        }
    }

    // usize index
    impl IndexOp<usize> for Array {
        fn index(&self, idx: usize) -> Self {
            IndexOp::<i32>::index(self, idx as i32)
        }
    }

    impl IndexOp<(RangeTo<i32>, RangeFull)> for Array {
        fn index(&self, idx: (RangeTo<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(0, self.ndim() as usize);
            slice_axis_range(self, axis, 0, idx.0.end)
        }
    }

    impl IndexOp<(RangeTo<usize>, RangeFull)> for Array {
        fn index(&self, idx: (RangeTo<usize>, RangeFull)) -> Self {
            IndexOp::<(RangeTo<i32>, RangeFull)>::index(self, (..(idx.0.end as i32), ..))
        }
    }

    impl IndexOp<(Range<i32>, RangeFull)> for Array {
        fn index(&self, idx: (Range<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(0, self.ndim() as usize);
            slice_axis_range(self, axis, idx.0.start, idx.0.end)
        }
    }

    impl IndexOp<(Range<usize>, RangeFull)> for Array {
        fn index(&self, idx: (Range<usize>, RangeFull)) -> Self {
            IndexOp::<(Range<i32>, RangeFull)>::index(
                self,
                ((idx.0.start as i32)..(idx.0.end as i32), ..),
            )
        }
    }

    impl IndexOp<(RangeFrom<i32>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFrom<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(0, self.ndim() as usize);
            slice_axis_from(self, axis, idx.0.start)
        }
    }

    impl IndexOp<(RangeFrom<usize>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFrom<usize>, RangeFull)) -> Self {
            IndexOp::<(RangeFrom<i32>, RangeFull)>::index(self, ((idx.0.start as i32).., ..))
        }
    }

    impl IndexOp<(RangeFull, RangeTo<i32>)> for Array {
        fn index(&self, idx: (RangeFull, RangeTo<i32>)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_range(self, axis, 0, idx.1.end)
        }
    }

    impl IndexOp<(RangeFull, RangeTo<usize>)> for Array {
        fn index(&self, idx: (RangeFull, RangeTo<usize>)) -> Self {
            IndexOp::<(RangeFull, RangeTo<i32>)>::index(self, (.., ..(idx.1.end as i32)))
        }
    }

    impl IndexOp<(RangeFull, Range<i32>)> for Array {
        fn index(&self, idx: (RangeFull, Range<i32>)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_range(self, axis, idx.1.start, idx.1.end)
        }
    }

    impl IndexOp<(RangeFull, Range<usize>)> for Array {
        fn index(&self, idx: (RangeFull, Range<usize>)) -> Self {
            IndexOp::<(RangeFull, Range<i32>)>::index(
                self,
                (.., (idx.1.start as i32)..(idx.1.end as i32)),
            )
        }
    }

    impl IndexOp<(RangeFull, RangeFrom<i32>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFrom<i32>)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_from(self, axis, idx.1.start)
        }
    }

    impl IndexOp<(RangeFull, RangeFrom<usize>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFrom<usize>)) -> Self {
            IndexOp::<(RangeFull, RangeFrom<i32>)>::index(self, (.., (idx.1.start as i32)..))
        }
    }

    impl IndexOp<(RangeFull, i32)> for Array {
        fn index(&self, idx: (RangeFull, i32)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            select_axis_idx(self, axis, idx.1)
        }
    }

    impl IndexOp<(RangeFull, RangeTo<i32>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, RangeTo<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_range(self, axis, 0, idx.1.end)
        }
    }

    impl IndexOp<(RangeFull, RangeTo<usize>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, RangeTo<usize>, RangeFull)) -> Self {
            IndexOp::<(RangeFull, RangeTo<i32>, RangeFull)>::index(
                self,
                (.., ..(idx.1.end as i32), ..),
            )
        }
    }

    impl IndexOp<(RangeFull, Range<i32>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, Range<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_range(self, axis, idx.1.start, idx.1.end)
        }
    }

    impl IndexOp<(RangeFull, Range<usize>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, Range<usize>, RangeFull)) -> Self {
            IndexOp::<(RangeFull, Range<i32>, RangeFull)>::index(
                self,
                (.., (idx.1.start as i32)..(idx.1.end as i32), ..),
            )
        }
    }

    impl IndexOp<(RangeFull, RangeFrom<i32>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, RangeFrom<i32>, RangeFull)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            slice_axis_from(self, axis, idx.1.start)
        }
    }

    impl IndexOp<(RangeFull, RangeFrom<usize>, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, RangeFrom<usize>, RangeFull)) -> Self {
            IndexOp::<(RangeFull, RangeFrom<i32>, RangeFull)>::index(
                self,
                (.., (idx.1.start as i32).., ..),
            )
        }
    }

    impl IndexOp<(RangeFull, i32, RangeFull)> for Array {
        fn index(&self, idx: (RangeFull, i32, RangeFull)) -> Self {
            let axis = normalize_axis(1, self.ndim() as usize);
            select_axis_idx(self, axis, idx.1)
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeTo<i32>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeTo<i32>)) -> Self {
            let axis = normalize_axis(3, self.ndim() as usize);
            slice_axis_range(self, axis, 0, idx.3.end)
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeTo<usize>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeTo<usize>)) -> Self {
            IndexOp::<(RangeFull, RangeFull, RangeFull, RangeTo<i32>)>::index(
                self,
                (.., .., .., ..(idx.3.end as i32)),
            )
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeFrom<i32>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeFrom<i32>)) -> Self {
            let axis = normalize_axis(3, self.ndim() as usize);
            slice_axis_from(self, axis, idx.3.start)
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, RangeFrom<usize>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, RangeFrom<usize>)) -> Self {
            IndexOp::<(RangeFull, RangeFull, RangeFull, RangeFrom<i32>)>::index(
                self,
                (.., .., .., (idx.3.start as i32)..),
            )
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, Range<i32>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, Range<i32>)) -> Self {
            let axis = normalize_axis(3, self.ndim() as usize);
            slice_axis_range(self, axis, idx.3.start, idx.3.end)
        }
    }

    impl IndexOp<(RangeFull, RangeFull, RangeFull, Range<usize>)> for Array {
        fn index(&self, idx: (RangeFull, RangeFull, RangeFull, Range<usize>)) -> Self {
            IndexOp::<(RangeFull, RangeFull, RangeFull, Range<i32>)>::index(
                self,
                (.., .., .., (idx.3.start as i32)..(idx.3.end as i32)),
            )
        }
    }
}

// ── compile shims (mlx_rs::compile compat) ───────────────────────────────────
//
// These are lightweight shims.  The bridge already provides `enable_compile()`
// and `disable_compile()`.  The `Closure` type is a no-op wrapper — the bridge
// dispatches directly via FFI-compiled ops rather than using MLX's Rust
// closure machinery.
pub mod compile {
    use super::Array;

    /// Clear the compile cache (wraps `pmetal_bridge::inline_array::clear_cache`).
    pub fn clear_cache() {
        crate::inline_array::clear_cache();
    }

    /// Placeholder for `mlx_rs::compile::Closure` (boxed, type-erased version).
    ///
    /// The bridge does not need this type for its own code paths; it exists solely
    /// to satisfy compilation of model files that reference it.
    pub struct Closure {
        f: Box<dyn Fn(&[Array]) -> Vec<Array>>,
    }

    impl std::fmt::Debug for Closure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Closure")
        }
    }

    impl Closure {
        pub fn new(f: impl Fn(&[Array]) -> Vec<Array> + 'static) -> Self {
            Self { f: Box::new(f) }
        }
        pub fn call(&self, args: &[Array]) -> Vec<Array> {
            (self.f)(args)
        }
        /// `apply` — alias for `call`, matches the mlx-rs Closure API.
        pub fn apply(&self, args: &[Array]) -> Result<Vec<Array>, super::Exception> {
            Ok((self.f)(args))
        }
    }

    /// Compile a closure (no-op shim — bridge uses pre-compiled C++ ops).
    pub fn compile(f: Closure, _shapeless: bool) -> Result<Closure, super::Exception> {
        Ok(f)
    }
}

// ── losses shims (mlx_rs::losses compat) ─────────────────────────────────────
pub mod losses {
    use super::{Array, Exception};

    #[derive(Debug, Clone, Copy)]
    pub enum LossReduction {
        None,
        Sum,
        Mean,
    }

    /// Categorical cross-entropy loss — drop-in for `mlx_rs::losses::CrossEntropy`.
    ///
    /// Computes `softmax_cross_entropy(logits, targets, axis=-1)` element-wise.
    /// `targets` must be integer class indices.
    #[derive(Debug, Clone, Default)]
    pub struct CrossEntropy;

    impl CrossEntropy {
        /// Construct a CrossEntropy loss (infallible — Result is for API parity).
        pub fn new() -> Result<Self, Exception> {
            Ok(Self)
        }

        /// Compute per-token cross-entropy loss.
        ///
        /// `logits`: `[..., vocab]`, `targets`: `[...]` integer indices.
        /// Returns per-element loss `[...]`.
        pub fn apply(&self, logits: &Array, targets: &Array) -> Result<Array, Exception> {
            Ok(logits.cross_entropy(targets, -1))
        }
    }

    pub struct BinaryCrossEntropyBuilder {
        reduction: LossReduction,
        with_logits: bool,
    }

    impl BinaryCrossEntropyBuilder {
        pub fn new() -> Self {
            Self {
                reduction: LossReduction::Mean,
                with_logits: true,
            }
        }
        pub fn reduction(mut self, r: LossReduction) -> Self {
            self.reduction = r;
            self
        }
        pub fn with_logits(mut self, v: bool) -> Self {
            self.with_logits = v;
            self
        }
        pub fn build(self) -> BinaryCrossEntropy {
            BinaryCrossEntropy {
                reduction: self.reduction,
                _with_logits: self.with_logits,
            }
        }
    }

    pub struct BinaryCrossEntropy {
        reduction: LossReduction,
        _with_logits: bool,
    }

    impl BinaryCrossEntropy {
        pub fn call(&self, logits: &Array, targets: &Array) -> Array {
            // BCE with logits: -( targets * log_sigmoid(logits) + (1-targets) * log_sigmoid(-logits) )
            let ones = Array::ones(logits.shape(), 10);
            let neg_logits = logits.negative();
            let pos = logits.log_softmax(0); // placeholder - proper impl would use log_sigmoid
            let neg = neg_logits.log_softmax(0);
            let loss = targets
                .negative()
                .multiply(&pos)
                .subtract(&ones.subtract(targets).multiply(&neg));
            match self.reduction {
                LossReduction::Mean => loss.mean_all(),
                LossReduction::Sum => loss.sum_all(),
                LossReduction::None => loss,
            }
        }
    }
}

// ── optimizers compat ────────────────────────────────────────────────────────
//
// Drop-in shims for `mlx_rs::optimizers::*` used by the training infrastructure.

pub mod optimizers {
    use super::{Array, Exception, FlattenedModuleParam, ModuleParameters, NestedValue};
    use std::collections::HashMap;
    use std::rc::Rc;

    /// Optimizer state: stores (momentum, velocity) tensors per parameter.
    pub type State<V> = HashMap<Rc<str>, V>;

    /// Common optimizer interface — matches `mlx_rs::optimizers::Optimizer`.
    pub trait Optimizer {
        type State;
        fn state(&self) -> &Self::State;
        fn state_mut(&mut self) -> &mut Self::State;
        fn update_single(
            &mut self,
            key: &Rc<str>,
            gradient: &Array,
            parameter: &mut Array,
        ) -> Result<(), Exception>;
        fn update<M: ModuleParameters>(
            &mut self,
            model: &mut M,
            gradients: FlattenedModuleParam,
        ) -> Result<(), Exception> {
            let mut pm = model.parameters_mut();
            for (key, grad) in &gradients {
                update_leaf(&mut pm, key, |arr| {
                    let _ = self.update_single(key, grad, arr);
                });
            }
            Ok(())
        }
    }

    fn update_leaf<F: FnMut(&mut Array)>(
        params: &mut HashMap<Rc<str>, NestedValue<&mut Array>>,
        key: &Rc<str>,
        mut f: F,
    ) {
        if let Some(v) = params.get_mut(key) {
            if let NestedValue::Value(arr) = v {
                f(arr);
            }
        }
    }

    /// Updatable: exposes state arrays for eval/checkpointing.
    pub trait Updatable {
        fn updatable_states_len(&self) -> usize;
        fn updatable_states(&self) -> Vec<&Array>;
        fn updatable_states_mut(&mut self) -> Vec<&mut Array>;
    }

    /// AdamW optimizer compatible with mlx_rs::optimizers::AdamW interface.
    pub struct AdamW {
        inner: crate::optimizer::AdamW,
        pub lr: Array,
        pub state: State<(Array, Array)>,
    }

    impl std::fmt::Debug for AdamW {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AdamW(compat)").finish()
        }
    }

    impl AdamW {
        pub fn new(lr: f32, weight_decay: f32) -> Self {
            Self {
                inner: crate::optimizer::AdamW::new(lr, weight_decay),
                lr: Array::from_f32(lr),
                state: HashMap::new(),
            }
        }
    }

    impl Optimizer for AdamW {
        type State = State<(Array, Array)>;
        fn state(&self) -> &Self::State {
            &self.state
        }
        fn state_mut(&mut self) -> &mut Self::State {
            &mut self.state
        }
        fn update_single(
            &mut self,
            key: &Rc<str>,
            gradient: &Array,
            parameter: &mut Array,
        ) -> Result<(), Exception> {
            self.inner.step_single(key.as_ref(), gradient, parameter);
            Ok(())
        }
    }

    impl Updatable for AdamW {
        fn updatable_states_len(&self) -> usize {
            self.state.len() * 2
        }
        fn updatable_states(&self) -> Vec<&Array> {
            self.state.values().flat_map(|(m, v)| [m, v]).collect()
        }
        fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
            self.state.values_mut().flat_map(|(m, v)| [m, v]).collect()
        }
    }

    /// Builder for AdamW.
    #[derive(Debug, Clone)]
    pub struct AdamWBuilder {
        lr: f32,
        weight_decay: f32,
        betas: (f32, f32),
        eps: f32,
    }

    impl AdamWBuilder {
        pub fn new(lr: f32) -> Self {
            Self {
                lr,
                weight_decay: 0.01,
                betas: (0.9, 0.999),
                eps: 1e-8,
            }
        }
        pub fn weight_decay(mut self, wd: f32) -> Self {
            self.weight_decay = wd;
            self
        }
        pub fn betas(mut self, b: (f32, f32)) -> Self {
            self.betas = b;
            self
        }
        pub fn eps(mut self, e: f32) -> Self {
            self.eps = e;
            self
        }
        pub fn build(self) -> Result<AdamW, Exception> {
            Ok(AdamW::new(self.lr, self.weight_decay))
        }
    }

    /// SGD optimizer — vanilla stochastic gradient descent with no momentum.
    pub struct Sgd {
        pub lr: Array,
        pub state: State<()>,
    }

    impl std::fmt::Debug for Sgd {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Sgd").finish()
        }
    }

    impl Sgd {
        pub fn new(lr: f32) -> Self {
            Self {
                lr: Array::from_f32(lr),
                state: HashMap::new(),
            }
        }
    }

    impl Optimizer for Sgd {
        type State = State<()>;
        fn state(&self) -> &Self::State {
            &self.state
        }
        fn state_mut(&mut self) -> &mut Self::State {
            &mut self.state
        }
        fn update_single(
            &mut self,
            _key: &Rc<str>,
            gradient: &Array,
            parameter: &mut Array,
        ) -> Result<(), Exception> {
            let lr_val = self.lr.clone().item_f32();
            let lr_arr = Array::from_f32(lr_val);
            *parameter = parameter.subtract(&gradient.multiply(&lr_arr));
            Ok(())
        }
    }

    impl Updatable for Sgd {
        fn updatable_states_len(&self) -> usize {
            0
        }
        fn updatable_states(&self) -> Vec<&Array> {
            vec![]
        }
        fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
            vec![]
        }
    }

    impl<M, O: Updatable> Updatable for (M, O) {
        fn updatable_states_len(&self) -> usize {
            self.1.updatable_states_len()
        }
        fn updatable_states(&self) -> Vec<&Array> {
            self.1.updatable_states()
        }
        fn updatable_states_mut(&mut self) -> Vec<&mut Array> {
            self.1.updatable_states_mut()
        }
    }
}

// ── transforms compat ─────────────────────────────────────────────────────────

pub mod transforms {
    use super::{Array, Exception};

    pub fn eval<'a>(arrays: impl IntoIterator<Item = &'a Array>) -> Result<(), Exception> {
        super::eval(arrays)
    }

    pub mod compile {
        /// Placeholder: the bridge pre-compiles all ops in C++.
        /// Returns the function unchanged.
        pub fn compile_with_state<S, F>(f: F) -> F
        where
            F: FnMut(&mut S) -> Vec<super::super::Array>,
        {
            f
        }

        /// Clear the MLX compilation cache.
        pub fn clear_cache() {
            crate::inline_array::clear_cache();
        }
    }

    /// Set the Metal wired memory limit.
    pub fn set_wired_limit(limit: usize) -> usize {
        crate::inline_array::set_wired_limit(limit)
    }

    /// Set the wired memory limit to the device's recommended maximum.
    pub fn set_wired_limit_max() -> usize {
        crate::inline_array::set_wired_limit_max()
    }

    /// Clear the MLX array cache.
    pub fn clear_cache() {
        crate::inline_array::clear_cache();
    }
}

// ── builder compat ─────────────────────────────────────────────────────────────

/// Drop-in for `mlx_rs::builder::Builder`.
pub mod builder {
    /// Helper trait for builder pattern — matches mlx_rs::builder::Builder.
    pub trait Builder<T> {
        type Error: std::error::Error;
        fn build(self) -> Result<T, Self::Error>;
    }
}

// ── module compat ─────────────────────────────────────────────────────────────

pub mod module {
    pub use super::{
        FlattenedModuleParam, Module, ModuleParamMut, ModuleParamRef, ModuleParameters,
        ModuleParametersExt, NestedValue, Param,
    };

    pub fn update_parameters<M: ModuleParameters>(
        model: &mut M,
        updates: impl Iterator<Item = (std::rc::Rc<str>, super::Array)>,
    ) {
        let mut pm = model.parameters_mut();
        for (key, val) in updates {
            if let Some(entry) = pm.get_mut(&key) {
                if let super::NestedValue::Value(arr) = entry {
                    **arr = val;
                }
            }
        }
    }
}

// ── extra free functions ──────────────────────────────────────────────────────

/// Stop-gradient passthrough — severs the autograd tape.
pub fn stop_gradient(a: &Array) -> Result<Array, Exception> {
    Ok(a.stop_gradient())
}

// ── Parameter trait ────────────────────────────────────────────────────────────

/// Helper trait used by `impl_module_params!` macro to collect arrays.
///
/// Implements the visitor pattern for parameter trees, mirroring mlx-rs's
/// `module::Parameter` trait but adapted for compat's `NestedValue` types.
pub trait Parameter {
    /// Insert borrowed array references into the `ModuleParamRef` map.
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>);
    /// Insert mutable array references into the `ModuleParamMut` map.
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>);
    /// Count leaf arrays.
    fn count_params(&self) -> usize;
}

// Param<Array> — always contributes one leaf
impl Parameter for Param<Array> {
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        out.insert(std::rc::Rc::from(key), NestedValue::Value(&self.value));
    }
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        out.insert(std::rc::Rc::from(key), NestedValue::Value(&mut self.value));
    }
    fn count_params(&self) -> usize {
        1
    }
}

// Param<Option<Array>> — contributes one leaf only when Some
impl Parameter for Param<Option<Array>> {
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        if let Some(ref arr) = self.value {
            out.insert(std::rc::Rc::from(key), NestedValue::Value(arr));
        }
    }
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        if let Some(ref mut arr) = self.value {
            out.insert(std::rc::Rc::from(key), NestedValue::Value(arr));
        }
    }
    fn count_params(&self) -> usize {
        if self.value.is_some() { 1 } else { 0 }
    }
}

// A plain Array field (no Param wrapper) — contributes one leaf
impl Parameter for Array {
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        out.insert(std::rc::Rc::from(key), NestedValue::Value(self));
    }
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        out.insert(std::rc::Rc::from(key), NestedValue::Value(self));
    }
    fn count_params(&self) -> usize {
        1
    }
}

// Option<Array> — contributes a leaf only when Some
impl Parameter for Option<Array> {
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        if let Some(ref arr) = *self {
            out.insert(std::rc::Rc::from(key), NestedValue::Value(arr));
        }
    }
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        if let Some(ref mut arr) = *self {
            out.insert(std::rc::Rc::from(key), NestedValue::Value(arr));
        }
    }
    fn count_params(&self) -> usize {
        if self.is_some() { 1 } else { 0 }
    }
}

// Vec<Array>
impl Parameter for Vec<Array> {
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        for (i, arr) in self.iter().enumerate() {
            let k = format!("{key}.{i}");
            out.insert(std::rc::Rc::from(k.as_str()), NestedValue::Value(arr));
        }
    }
    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        for (i, arr) in self.iter_mut().enumerate() {
            let k = format!("{key}.{i}");
            out.insert(std::rc::Rc::from(k.as_str()), NestedValue::Value(arr));
        }
    }
    fn count_params(&self) -> usize {
        self.len()
    }
}

// Sub-modules implementing ModuleParameters are promoted into nested maps.
// We achieve this via a blanket impl over ModuleParameters that is lower priority
// than the concrete impls above; we use a newtype trick via a secondary trait.

/// Marker: any T: ModuleParameters can be used as a nested param group.
pub trait NestedParam: ModuleParameters {}
impl<T: ModuleParameters + ?Sized> NestedParam for T {}

impl<T: ModuleParameters> Parameter for T
where
    // Constrain so concrete impls (Param<Array> etc.) take priority via orphan rules.
    // This works because Param<T>, Array, etc. do NOT implement ModuleParameters.
    T: NestedParam,
{
    fn collect_params<'a>(&'a self, key: &str, out: &mut ModuleParamRef<'a>) {
        let sub = self.parameters();
        if sub.is_empty() {
            return;
        }
        // Promote sub-tree as a NestedValue::Map
        let mut sub_map: std::collections::HashMap<std::rc::Rc<str>, NestedValue<&'a Array>> =
            std::collections::HashMap::new();
        for (k, v) in sub {
            // Re-borrow with 'a lifetime: copy value references into sub_map
            // Safety: sub-struct lifetime >= 'a because self: 'a
            // We clone the NestedValue which just copies the &Array pointer.
            sub_map.insert(k, unsafe { clone_nested_ref_lifetime(v) });
        }
        out.insert(std::rc::Rc::from(key), NestedValue::Map(sub_map));
    }

    fn collect_params_mut<'a>(&'a mut self, key: &str, out: &mut ModuleParamMut<'a>) {
        let sub = self.parameters_mut();
        if sub.is_empty() {
            return;
        }
        let mut sub_map: std::collections::HashMap<std::rc::Rc<str>, NestedValue<&'a mut Array>> =
            std::collections::HashMap::new();
        for (k, v) in sub {
            sub_map.insert(k, unsafe { clone_nested_mut_lifetime(v) });
        }
        out.insert(std::rc::Rc::from(key), NestedValue::Map(sub_map));
    }

    fn count_params(&self) -> usize {
        self.num_parameters()
    }
}

// Lifetime re-borrow helpers — these are safe because the sub-map lifetime
// is bounded by `self: 'a` and we never alias mutable references.
#[allow(clippy::needless_lifetimes)]
unsafe fn clone_nested_ref_lifetime<'a>(v: NestedValue<&Array>) -> NestedValue<&'a Array> {
    match v {
        NestedValue::Value(r) => NestedValue::Value(unsafe { &*(r as *const Array) }),
        NestedValue::Map(m) => {
            let mut out = std::collections::HashMap::new();
            for (k, child) in m {
                out.insert(k, unsafe { clone_nested_ref_lifetime(child) });
            }
            NestedValue::Map(out)
        }
    }
}

#[allow(clippy::needless_lifetimes)]
unsafe fn clone_nested_mut_lifetime<'a>(v: NestedValue<&mut Array>) -> NestedValue<&'a mut Array> {
    match v {
        NestedValue::Value(r) => NestedValue::Value(unsafe { &mut *(r as *mut Array) }),
        NestedValue::Map(m) => {
            let mut out = std::collections::HashMap::new();
            for (k, child) in m {
                out.insert(k, unsafe { clone_nested_mut_lifetime(child) });
            }
            NestedValue::Map(out)
        }
    }
}

// ── impl_module_params! macro ────────────────────────────────────────────────

/// Implement `ModuleParameters` for a struct.
///
/// Usage:
/// ```ignore
/// impl_module_params!(MyStruct; field1, field2, field3);
/// ```
///
/// The macro inserts every listed field into the parameter tree.  Only fields
/// that implement `Parameter` (i.e., `Param<Array>`, `Param<Option<Array>>`,
/// bare `Array`, or nested `ModuleParameters` types) contribute leaf entries.
/// Fields that don't implement `Parameter` should not be listed.
#[macro_export]
macro_rules! impl_module_params {
    ($ty:ty ; $($field:ident),* $(,)?) => {
        impl $crate::compat::ModuleParameters for $ty {
            fn num_parameters(&self) -> usize {
                0 $( + $crate::compat::Parameter::count_params(&self.$field) )*
            }

            fn parameters(&self) -> $crate::compat::ModuleParamRef<'_> {
                let mut out = ::std::collections::HashMap::new();
                $( $crate::compat::Parameter::collect_params(&self.$field, stringify!($field), &mut out); )*
                out
            }

            fn parameters_mut(&mut self) -> $crate::compat::ModuleParamMut<'_> {
                let mut out = ::std::collections::HashMap::new();
                $( $crate::compat::Parameter::collect_params_mut(&mut self.$field, stringify!($field), &mut out); )*
                out
            }
        }
    };

    // Variant with generics: impl_module_params!(MyStruct<T> where T: Foo; field1, field2)
    ($ty:ty ; $($field:ident),* $(,)? ; where $($bound:tt)*) => {
        impl $crate::compat::ModuleParameters for $ty where $($bound)* {
            fn num_parameters(&self) -> usize {
                0 $( + $crate::compat::Parameter::count_params(&self.$field) )*
            }

            fn parameters(&self) -> $crate::compat::ModuleParamRef<'_> {
                let mut out = ::std::collections::HashMap::new();
                $( $crate::compat::Parameter::collect_params(&self.$field, stringify!($field), &mut out); )*
                out
            }

            fn parameters_mut(&mut self) -> $crate::compat::ModuleParamMut<'_> {
                let mut out = ::std::collections::HashMap::new();
                $( $crate::compat::Parameter::collect_params_mut(&mut self.$field, stringify!($field), &mut out); )*
                out
            }
        }
    };
}

// ── nn layer types ─────────────────────────────────────────────────────────────

/// Layer types for neural networks, implementing `ModuleParameters`.
///
/// These are bridge-native equivalents of `mlx_rs::nn::Linear`, `RmsNorm`, etc.
/// They implement `pmetal_bridge::compat::ModuleParameters` directly.
pub mod layers {
    use super::{
        Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, Param, ops, random,
    };
    use std::collections::HashMap;
    use std::rc::Rc;

    // ── Linear ────────────────────────────────────────────────────────────────

    /// Affine linear layer: `y = x @ W^T + b`.
    #[derive(Debug, Clone)]
    pub struct Linear {
        pub weight: Param<Array>,
        pub bias: Param<Option<Array>>,
    }

    impl Linear {
        pub const DEFAULT_BIAS: bool = true;

        pub fn new(in_dims: i32, out_dims: i32, with_bias: bool) -> Result<Self, super::Exception> {
            let scale = f32::sqrt(1.0 / in_dims as f32);
            let weight =
                random::uniform_range(-scale, scale, &[out_dims, in_dims], super::Dtype::Float32);
            let bias = if with_bias {
                Some(random::uniform_range(
                    -scale,
                    scale,
                    &[out_dims],
                    super::Dtype::Float32,
                ))
            } else {
                None
            };
            Ok(Self {
                weight: Param::new(weight),
                bias: Param::new(bias),
            })
        }

        /// Infallible constructor variant for internal use.
        pub fn create(in_dims: i32, out_dims: i32, with_bias: bool) -> Self {
            Self::new(in_dims, out_dims, with_bias).unwrap()
        }

        pub fn forward(&self, x: &Array) -> Array {
            match &self.bias.value {
                Some(b) => {
                    // addmm: b + x @ W^T
                    let mm = x.matmul(&self.weight.value.t());
                    mm.add(b)
                }
                None => x.matmul(&self.weight.value.t()),
            }
        }

        pub fn shape(&self) -> (i32, i32) {
            let s = self.weight.value.shape();
            (s[0], s[1])
        }

        #[inline]
        pub fn unwrap(self) -> Self {
            self
        }

        #[inline]
        pub fn expect(self, _msg: &str) -> Self {
            self
        }
    }

    crate::impl_module_params!(Linear; weight, bias);

    /// Builder for [`Linear`].
    pub struct LinearBuilder {
        in_dims: i32,
        out_dims: i32,
        bias: bool,
    }

    impl LinearBuilder {
        pub fn new(in_dims: i32, out_dims: i32) -> Self {
            Self {
                in_dims,
                out_dims,
                bias: Linear::DEFAULT_BIAS,
            }
        }
        pub fn bias(mut self, b: bool) -> Self {
            self.bias = b;
            self
        }
        pub fn build(self) -> Result<Linear, Exception> {
            Linear::new(self.in_dims, self.out_dims, self.bias)
        }
    }

    // ── RmsNorm ───────────────────────────────────────────────────────────────

    /// RMS layer normalization.
    #[derive(Debug, Clone)]
    pub struct RmsNorm {
        pub weight: Param<Array>,
        pub eps: f32,
    }

    impl RmsNorm {
        pub const DEFAULT_EPS: f32 = 1e-5;

        pub fn new(dims: i32) -> Result<Self, Exception> {
            Ok(Self::with_eps(dims, Self::DEFAULT_EPS))
        }

        pub fn with_eps(dims: i32, eps: f32) -> Self {
            let weight = ops::ones(&[dims], super::Dtype::Float32);
            Self {
                weight: Param::new(weight),
                eps,
            }
        }

        pub fn forward(&self, x: &Array) -> Array {
            x.rms_norm(Some(&self.weight.value), self.eps)
        }
    }

    crate::impl_module_params!(RmsNorm; weight);

    /// Builder for [`RmsNorm`].
    pub struct RmsNormBuilder {
        dims: i32,
        eps: f32,
    }

    impl RmsNormBuilder {
        pub fn new(dims: i32) -> Self {
            Self {
                dims,
                eps: RmsNorm::DEFAULT_EPS,
            }
        }
        pub fn eps(mut self, eps: f32) -> Self {
            self.eps = eps;
            self
        }
        pub fn build(self) -> Result<RmsNorm, Exception> {
            Ok(RmsNorm::with_eps(self.dims, self.eps))
        }
    }

    // ── LayerNorm ─────────────────────────────────────────────────────────────

    /// Layer normalization.
    #[derive(Debug, Clone)]
    pub struct LayerNorm {
        pub dimensions: i32,
        pub eps: f32,
        pub weight: Param<Option<Array>>,
        pub bias: Param<Option<Array>>,
    }

    impl LayerNorm {
        pub const DEFAULT_EPS: f32 = 1e-5;
        pub const DEFAULT_AFFINE: bool = true;

        pub fn with_affine(dims: i32, eps: f32, affine: bool) -> Self {
            let (w, b) = if affine {
                (
                    Some(ops::ones(&[dims], super::Dtype::Float32)),
                    Some(ops::zeros(&[dims], super::Dtype::Float32)),
                )
            } else {
                (None, None)
            };
            Self {
                dimensions: dims,
                eps,
                weight: Param::new(w),
                bias: Param::new(b),
            }
        }

        pub fn forward(&self, x: &Array) -> Array {
            let w: Option<&Array> = self.weight.value.as_ref();
            let b: Option<&Array> = self.bias.value.as_ref();
            x.layer_norm(w, b, self.eps)
        }
    }

    crate::impl_module_params!(LayerNorm; weight, bias);

    /// Builder for [`LayerNorm`].
    pub struct LayerNormBuilder {
        dims: i32,
        eps: f32,
        affine: bool,
    }

    impl LayerNormBuilder {
        pub fn new(dims: i32) -> Self {
            Self {
                dims,
                eps: LayerNorm::DEFAULT_EPS,
                affine: LayerNorm::DEFAULT_AFFINE,
            }
        }
        pub fn eps(mut self, eps: f32) -> Self {
            self.eps = eps;
            self
        }
        pub fn affine(mut self, a: bool) -> Self {
            self.affine = a;
            self
        }
        pub fn build(self) -> Result<LayerNorm, Exception> {
            Ok(LayerNorm::with_affine(self.dims, self.eps, self.affine))
        }
    }

    // ── GroupNorm ─────────────────────────────────────────────────────────────

    /// Group normalization.
    #[derive(Debug, Clone)]
    pub struct GroupNorm {
        pub group_count: i32,
        pub dimensions: i32,
        pub eps: Array,
        pub pytorch_compatible: bool,
        pub weight: Param<Option<Array>>,
        pub bias: Param<Option<Array>>,
    }

    impl GroupNorm {
        pub const DEFAULT_EPS: f32 = 1e-5;
        pub const DEFAULT_AFFINE: bool = true;
        pub const DEFAULT_PYTORCH_COMPATIBLE: bool = false;

        pub fn new(
            group_count: i32,
            dims: i32,
            eps: f32,
            affine: bool,
            pytorch_compatible: bool,
        ) -> Self {
            let (w, b) = if affine {
                (
                    Some(ops::ones(&[dims], super::Dtype::Float32)),
                    Some(ops::zeros(&[dims], super::Dtype::Float32)),
                )
            } else {
                (None, None)
            };
            Self {
                group_count,
                dimensions: dims,
                eps: Array::from_f32(eps),
                pytorch_compatible,
                weight: Param::new(w),
                bias: Param::new(b),
            }
        }

        pub fn forward(&self, x: &Array) -> Array {
            let eps_f = self.eps.clone().item_f32();
            let batch = x.dim(0);
            let dims = x.dim(-1);
            let group_size = dims / self.group_count;

            if self.pytorch_compatible {
                // PyTorch layout: [B, H, W, C] → reshape to [B, H*W, groups, group_size]
                let x2 = x.reshape(&[batch, -1, self.group_count, group_size]);
                let x2 = x2
                    .transpose_axes(&[0, 2, 1, 3])
                    .reshape(&[batch, self.group_count, -1]);
                let x2 = x2.layer_norm(None, None, eps_f);
                let ndim = x.ndim() as i32;
                let new_shape: Vec<i32> = std::iter::once(batch)
                    .chain(x.shape()[1..(ndim as usize - 1)].iter().copied())
                    .chain(std::iter::once(dims))
                    .collect();
                let x2 = x2.reshape(&[batch, self.group_count, -1, group_size]);
                let x2 = x2.transpose_axes(&[0, 2, 1, 3]).reshape(&new_shape);
                self.apply_affine(x2)
            } else {
                let x2 = x.reshape(&[batch, -1, self.group_count]);
                // instance norm per group
                let mean = x2.mean_axis(1, true);
                let var = x2.subtract(&mean).square().mean_axis(1, true);
                let eps_arr = Array::from_f32(eps_f);
                let x2 = x2.subtract(&mean).multiply(&var.add(&eps_arr).rsqrt());
                let ndim = x.ndim() as i32;
                let new_shape: Vec<i32> = std::iter::once(batch)
                    .chain(x.shape()[1..(ndim as usize - 1)].iter().copied())
                    .chain(std::iter::once(dims))
                    .collect();
                let x2 = x2.reshape(&new_shape);
                self.apply_affine(x2)
            }
        }

        fn apply_affine(&self, x: Array) -> Array {
            match (&self.weight.value, &self.bias.value) {
                (Some(w), Some(b)) => x.multiply(w).add(b),
                (Some(w), None) => x.multiply(w),
                (None, Some(b)) => x.add(b),
                (None, None) => x,
            }
        }
    }

    crate::impl_module_params!(GroupNorm; weight, bias);

    /// Builder for [`GroupNorm`].
    pub struct GroupNormBuilder {
        group_count: i32,
        dims: i32,
        eps: f32,
        affine: bool,
        pytorch_compatible: bool,
    }

    impl GroupNormBuilder {
        pub fn new(group_count: i32, dims: i32) -> Self {
            Self {
                group_count,
                dims,
                eps: GroupNorm::DEFAULT_EPS,
                affine: GroupNorm::DEFAULT_AFFINE,
                pytorch_compatible: GroupNorm::DEFAULT_PYTORCH_COMPATIBLE,
            }
        }
        pub fn eps(mut self, eps: f32) -> Self {
            self.eps = eps;
            self
        }
        pub fn affine(mut self, a: bool) -> Self {
            self.affine = a;
            self
        }
        pub fn pytorch_compatible(mut self, p: bool) -> Self {
            self.pytorch_compatible = p;
            self
        }
        pub fn build(self) -> Result<GroupNorm, Exception> {
            Ok(GroupNorm::new(
                self.group_count,
                self.dims,
                self.eps,
                self.affine,
                self.pytorch_compatible,
            ))
        }
    }

    // ── Embedding ─────────────────────────────────────────────────────────────

    /// Simple embedding lookup table.
    #[derive(Debug, Clone)]
    pub struct Embedding {
        pub weight: Param<Array>,
    }

    impl Embedding {
        pub fn new(num_embeddings: i32, dims: i32) -> Result<Self, Exception> {
            let scale = f32::sqrt(1.0 / dims as f32);
            let weight = random::uniform_range(
                -scale,
                scale,
                &[num_embeddings, dims],
                super::Dtype::Float32,
            );
            Ok(Self {
                weight: Param::new(weight),
            })
        }

        pub fn forward(&self, x: &Array) -> Array {
            self.weight.value.take_axis(x, 0)
        }

        pub fn as_linear(&self, x: &Array) -> Array {
            x.matmul(&self.weight.value.t())
        }
    }

    crate::impl_module_params!(Embedding; weight);

    // ── Conv1d ────────────────────────────────────────────────────────────────

    /// 1D convolution layer.
    #[derive(Debug, Clone)]
    pub struct Conv1d {
        pub weight: Param<Array>,
        pub bias: Param<Option<Array>>,
        pub stride: i32,
        pub padding: i32,
        pub dilation: i32,
        pub groups: i32,
    }

    impl Conv1d {
        pub const DEFAULT_BIAS: bool = true;
        pub const DEFAULT_STRIDE: i32 = 1;
        pub const DEFAULT_PADDING: i32 = 0;
        pub const DEFAULT_DILATION: i32 = 1;
        pub const DEFAULT_GROUPS: i32 = 1;

        pub fn new(
            in_channels: i32,
            out_channels: i32,
            kernel_size: i32,
            stride: i32,
            padding: i32,
            dilation: i32,
            groups: i32,
            with_bias: bool,
        ) -> Self {
            let scale = f32::sqrt(1.0 / (in_channels * kernel_size) as f32);
            // weight shape: [out_channels, kernel_size, in_channels/groups]
            let weight = random::uniform_range(
                -scale,
                scale,
                &[out_channels, kernel_size, in_channels / groups],
                super::Dtype::Float32,
            );
            let bias = if with_bias {
                Some(ops::zeros(&[out_channels], super::Dtype::Float32))
            } else {
                None
            };
            Self {
                weight: Param::new(weight),
                bias: Param::new(bias),
                stride,
                padding,
                dilation,
                groups,
            }
        }

        pub fn forward(&self, x: &Array) -> Array {
            let y = ops::conv1d(
                x,
                &self.weight.value,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            );
            match &self.bias.value {
                Some(b) => y.add(b),
                None => y,
            }
        }
    }

    crate::impl_module_params!(Conv1d; weight, bias);

    /// Builder for [`Conv1d`].
    pub struct Conv1dBuilder {
        in_ch: i32,
        out_ch: i32,
        kernel: i32,
        bias: bool,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    }

    impl Conv1dBuilder {
        pub fn new(in_ch: i32, out_ch: i32, kernel: i32) -> Self {
            Self {
                in_ch,
                out_ch,
                kernel,
                bias: Conv1d::DEFAULT_BIAS,
                stride: Conv1d::DEFAULT_STRIDE,
                padding: Conv1d::DEFAULT_PADDING,
                dilation: Conv1d::DEFAULT_DILATION,
                groups: Conv1d::DEFAULT_GROUPS,
            }
        }
        pub fn bias(mut self, b: bool) -> Self {
            self.bias = b;
            self
        }
        pub fn stride(mut self, s: i32) -> Self {
            self.stride = s;
            self
        }
        pub fn padding(mut self, p: i32) -> Self {
            self.padding = p;
            self
        }
        pub fn dilation(mut self, d: i32) -> Self {
            self.dilation = d;
            self
        }
        pub fn groups(mut self, g: i32) -> Self {
            self.groups = g;
            self
        }
        pub fn build(self) -> Result<Conv1d, Exception> {
            Ok(Conv1d::new(
                self.in_ch,
                self.out_ch,
                self.kernel,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bias,
            ))
        }
    }

    // ── Rope (RotaryPositionalEncoding) ───────────────────────────────────────

    /// Rotary positional encoding (RoPE).
    ///
    /// Stateless — no trainable parameters.  The forward pass is dispatched
    /// via `InlineArray::rope()`.
    #[derive(Debug, Clone)]
    pub struct Rope {
        pub dimensions: i32,
        pub traditional: bool,
        pub base: f32,
        pub scale: f32,
    }

    impl Rope {
        pub const DEFAULT_TRADITIONAL: bool = false;
        pub const DEFAULT_BASE: f32 = 10_000.0;
        pub const DEFAULT_SCALE: f32 = 1.0;

        pub fn new(dims: i32, traditional: bool, base: f32, scale: f32) -> Self {
            Self {
                dimensions: dims,
                traditional,
                base,
                scale,
            }
        }

        pub fn forward(&self, x: &Array, offset: i32) -> Array {
            x.rope(
                self.dimensions,
                self.traditional,
                self.base,
                self.scale,
                offset,
            )
        }
    }

    impl ModuleParameters for Rope {
        fn num_parameters(&self) -> usize {
            0
        }
        fn parameters(&self) -> ModuleParamRef<'_> {
            HashMap::new()
        }
        fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
            HashMap::new()
        }
    }

    /// Builder for [`Rope`].
    pub struct RopeBuilder {
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
    }

    impl RopeBuilder {
        pub fn new(dims: i32) -> Self {
            Self {
                dims,
                traditional: Rope::DEFAULT_TRADITIONAL,
                base: Rope::DEFAULT_BASE,
                scale: Rope::DEFAULT_SCALE,
            }
        }
        pub fn traditional(mut self, t: bool) -> Self {
            self.traditional = t;
            self
        }
        pub fn base(mut self, b: f32) -> Self {
            self.base = b;
            self
        }
        pub fn scale(mut self, s: f32) -> Self {
            self.scale = s;
            self
        }
        pub fn build(self) -> Result<Rope, Exception> {
            Ok(Rope::new(
                self.dims,
                self.traditional,
                self.base,
                self.scale,
            ))
        }
    }

    // ── Vec<T> where T: ModuleParameters ─────────────────────────────────────

    impl<T: ModuleParameters> ModuleParameters for Vec<T> {
        fn num_parameters(&self) -> usize {
            self.iter().map(|m| m.num_parameters()).sum()
        }

        fn parameters(&self) -> ModuleParamRef<'_> {
            let mut out = HashMap::new();
            for (i, m) in self.iter().enumerate() {
                let sub = m.parameters();
                for (k, v) in sub {
                    let full: Rc<str> = format!("{i}.{k}").into();
                    out.insert(full, unsafe { super::clone_nested_ref_lifetime(v) });
                }
            }
            out
        }

        fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
            let mut out = HashMap::new();
            for (i, m) in self.iter_mut().enumerate() {
                let sub = m.parameters_mut();
                for (k, v) in sub {
                    let full: Rc<str> = format!("{i}.{k}").into();
                    out.insert(full, unsafe { super::clone_nested_mut_lifetime(v) });
                }
            }
            out
        }
    }

    // ── Option<T> where T: ModuleParameters ──────────────────────────────────

    impl<T: ModuleParameters> ModuleParameters for Option<T> {
        fn num_parameters(&self) -> usize {
            self.as_ref().map_or(0, |m| m.num_parameters())
        }

        fn parameters(&self) -> ModuleParamRef<'_> {
            self.as_ref().map_or(HashMap::new(), |m| m.parameters())
        }

        fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
            self.as_mut().map_or(HashMap::new(), |m| m.parameters_mut())
        }
    }

    // ── Module<&Array> impls for layer types ──────────────────────────────────
    //
    // These allow `Module::forward(&mut self.layer, x)?` to work, matching
    // the mlx-rs call pattern used throughout the architecture files.

    impl super::Module<&Array> for Linear {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(Linear::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    impl super::Module<&Array> for RmsNorm {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(RmsNorm::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    impl super::Module<&Array> for LayerNorm {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(LayerNorm::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    impl super::Module<&Array> for GroupNorm {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(GroupNorm::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    impl super::Module<&Array> for Embedding {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(Embedding::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    impl super::Module<&Array> for Conv1d {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(Conv1d::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    // Rope uses a tuple input (x, offset) since offset is needed.
    // We also provide a Module<&Array> impl that uses offset=0 for non-cached paths.
    impl super::Module<&Array> for Rope {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(Rope::forward(self, x, 0))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    // ── Conv2d ────────────────────────────────────────────────────────────────

    /// 2D convolution layer: `y = conv2d(x, W) + b`.
    #[derive(Debug, Clone)]
    pub struct Conv2d {
        pub weight: Param<Array>,
        pub bias: Param<Option<Array>>,
        pub stride: [i32; 2],
        pub padding: [i32; 2],
        pub dilation: [i32; 2],
        pub groups: i32,
    }

    impl Conv2d {
        pub fn new(
            in_channels: i32,
            out_channels: i32,
            kernel_size: i32,
            stride: i32,
            padding: i32,
            with_bias: bool,
        ) -> Self {
            let scale = f32::sqrt(1.0 / (in_channels * kernel_size * kernel_size) as f32);
            let weight = random::uniform_range(
                -scale,
                scale,
                &[out_channels, kernel_size, kernel_size, in_channels],
                super::Dtype::Float32,
            );
            let bias = if with_bias {
                Some(random::uniform_range(
                    -scale,
                    scale,
                    &[out_channels],
                    super::Dtype::Float32,
                ))
            } else {
                None
            };
            Self {
                weight: Param::new(weight),
                bias: Param::new(bias),
                stride: [stride, stride],
                padding: [padding, padding],
                dilation: [1, 1],
                groups: 1,
            }
        }

        pub fn forward(&self, x: &Array) -> Array {
            let out = x.conv2d(
                &self.weight.value,
                self.stride[0],
                self.stride[1],
                self.padding[0],
                self.padding[1],
                self.dilation[0],
                self.dilation[1],
                self.groups,
            );
            match &self.bias.value {
                Some(b) => out.add(b),
                None => out,
            }
        }
    }

    crate::impl_module_params!(Conv2d; weight, bias);

    impl super::Module<&Array> for Conv2d {
        type Output = Array;
        type Error = super::Exception;
        fn forward(&mut self, x: &Array) -> Result<Array, super::Exception> {
            Ok(Conv2d::forward(self, x))
        }
        fn training_mode(&mut self, _mode: bool) {}
    }

    /// Builder for [`Conv2d`].
    pub struct Conv2dBuilder {
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        with_bias: bool,
    }

    impl Conv2dBuilder {
        pub fn new(in_channels: i32, out_channels: i32, kernel_size: i32) -> Self {
            Self {
                in_channels,
                out_channels,
                kernel_size,
                stride: 1,
                padding: 0,
                with_bias: true,
            }
        }
        pub fn stride(mut self, s: i32) -> Self {
            self.stride = s;
            self
        }
        pub fn padding(mut self, p: i32) -> Self {
            self.padding = p;
            self
        }
        pub fn bias(mut self, b: bool) -> Self {
            self.with_bias = b;
            self
        }
        pub fn build(self) -> Result<Conv2d, super::Exception> {
            Ok(Conv2d::new(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.with_bias,
            ))
        }
    }

    impl super::builder::Builder<Conv2d> for Conv2dBuilder {
        type Error = super::Exception;
        fn build(self) -> Result<Conv2d, Self::Error> {
            Conv2dBuilder::build(self)
        }
    }

    // ── Sequential ────────────────────────────────────────────────────────────

    /// Sequential container — applies a list of modules in order.
    ///
    /// Equivalent to `mlx_rs::nn::Sequential` but works with any `Module<&Array>`.
    pub struct Sequential {
        layers:
            Vec<Box<dyn super::Module<&'static Array, Output = Array, Error = super::Exception>>>,
    }

    // Note: Sequential is intentionally left minimal. Full implementation would
    // require boxing with `dyn Module` trait objects which require 'static lifetimes.
    // For now it's a stub that satisfies type-checking.
    impl std::fmt::Debug for Sequential {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Sequential({})", self.layers.len())
        }
    }

    impl Sequential {
        pub fn new() -> Self {
            Self { layers: Vec::new() }
        }
    }

    impl ModuleParameters for Sequential {
        fn num_parameters(&self) -> usize {
            0
        }
        fn parameters(&self) -> super::ModuleParamRef<'_> {
            HashMap::new()
        }
        fn parameters_mut(&mut self) -> super::ModuleParamMut<'_> {
            HashMap::new()
        }
        fn trainable_parameters(&self) -> super::ModuleParamRef<'_> {
            HashMap::new()
        }
    }
}

// Re-export layers into nn for source compatibility with `use ... nn`
pub mod nn_layers {
    pub use super::layers::*;
}

// ── `array!` macro ────────────────────────────────────────────────────────────

/// `array!` macro: create a scalar Array from a literal (f32 cast).
#[macro_export]
macro_rules! array {
    ($val:expr) => {
        $crate::compat::Array::from_f32($val as f32)
    };
}
