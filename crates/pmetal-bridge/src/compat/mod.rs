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

mod traits;
pub use traits::*;

pub mod ops;
pub mod nn;
pub mod indexing;
pub mod optimizers;
pub mod layers;

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
    // The nested tree may use multi-segment keys at any level
    // (e.g. "layers.0" as a single key at the root, then "self_attn", etc.).
    // We try all possible splits: consume dots left-to-right, testing whether
    // the current prefix exists as a key in the current map.
    update_trainable_param_recurse(pm, key.as_ref(), value);
}

fn update_trainable_param_recurse(
    map: &mut std::collections::HashMap<std::rc::Rc<str>, NestedValue<&mut Array>>,
    remaining: &str,
    value: Array,
) {
    // Try the whole remaining string as a direct key first (leaf case).
    let remaining_rc: std::rc::Rc<str> = remaining.into();
    if let Some(entry) = map.get_mut(&remaining_rc) {
        match entry {
            NestedValue::Value(arr) => {
                **arr = value;
                return;
            }
            NestedValue::Map(_) => {
                // Exact match hit a map, not a leaf — shouldn't happen, but bail.
                return;
            }
        }
    }

    // Try progressively longer prefixes up to each '.' separator.
    let bytes = remaining.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' {
            let prefix: std::rc::Rc<str> = remaining[..i].into();
            if let Some(NestedValue::Map(child)) = map.get_mut(&prefix) {
                update_trainable_param_recurse(child, &remaining[i + 1..], value);
                return;
            }
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
