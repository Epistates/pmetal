use super::{
    Array, Exception, FlattenedModuleParam, ModuleParamMut, ModuleParamRef, ModuleParameters,
    NestedValue, Param,
};

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
        super::flatten_nested_ref_owned(&tree, "", &mut out);
        out
    }

    /// Flatten all mutable parameters into a `HashMap<String, &mut Array>`.
    ///
    /// Used by weight loaders that need to assign tensors by name.
    fn flatten_params_mut(&mut self) -> std::collections::HashMap<String, &mut Array> {
        let tree = self.parameters_mut();
        let mut out: std::collections::HashMap<String, &mut Array> =
            std::collections::HashMap::new();
        super::flatten_nested_mut_owned(tree, "", &mut out);
        out
    }

    /// Evaluate all parameters (materialise lazy computation graph).
    fn eval(&self) -> Result<(), Exception> {
        let p = self.parameters();
        super::eval_params(p)
    }
}

/// Blanket impl — any `ModuleParameters` gets `ModuleParametersExt` for free.
impl<T: ModuleParameters> ModuleParametersExt for T {}

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
            sub_map.insert(k, unsafe { super::clone_nested_ref_lifetime(v) });
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
            sub_map.insert(k, unsafe { super::clone_nested_mut_lifetime(v) });
        }
        out.insert(std::rc::Rc::from(key), NestedValue::Map(sub_map));
    }

    fn count_params(&self) -> usize {
        self.num_parameters()
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
