//! Weight normalisation (Salimans & Kingma, 2016) and checkpoint state dicts.
//!
//! BigVGAN trains every convolution under weight normalisation,
//! `W = g * v / ||v||` with the norm taken per output channel, and removes it
//! for inference by folding `W` into a plain weight. [`NormedWeight`] holds
//! either form behind one `materialize`, so a layer does not care which it has.

use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{
    Array, Dtype, ModuleParamMut, ModuleParamRef, ModuleParameters, NestedValue, Param,
};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// A convolution weight, weight-normalised or plain.
///
/// `weight` is the direction `v` when `magnitude` (`g`) is present and the
/// finished weight otherwise. The layout is the layer's own (channels-last);
/// the norm runs over every axis but the first.
#[derive(Debug, Clone)]
pub struct NormedWeight {
    /// Direction `v` (normalised form) or the weight itself (fused form).
    pub weight: Param<Array>,
    /// Per-output-channel magnitude `g`, shaped `[out, 1, ..., 1]`.
    pub magnitude: Option<Param<Array>>,
}

impl NormedWeight {
    /// Wrap a freshly initialised weight. With `weight_norm`, `g` starts at
    /// `||v||` so the layer's initial function is unchanged by the
    /// reparameterisation.
    pub fn new(weight: Array, weight_norm: bool) -> Self {
        let magnitude = weight_norm.then(|| Param::new(Self::norm_of(&weight)));
        Self {
            weight: Param::new(weight),
            magnitude,
        }
    }

    /// `||v||` per output channel, `[out, 1, ..., 1]`, in float32.
    fn norm_of(v: &Array) -> Array {
        let axes: Vec<i32> = (1..v.ndim()).collect();
        let v32 = v.as_dtype(Dtype::Float32);
        v32.multiply(&v32).sum_axes(&axes, true).sqrt()
    }

    /// Whether the weight is currently weight-normalised.
    pub fn is_normalized(&self) -> bool {
        self.magnitude.is_some()
    }

    /// The effective weight `g * v / ||v||` (or the plain weight), in `v`'s dtype.
    pub fn materialize(&self) -> Array {
        let v = &self.weight.value;
        match &self.magnitude {
            None => v.clone(),
            Some(g) => {
                let scale = g.value.as_dtype(Dtype::Float32).divide(&Self::norm_of(v));
                v.as_dtype(Dtype::Float32)
                    .multiply(&scale)
                    .as_dtype(v.dtype())
            }
        }
    }

    /// Fold the normalisation into a plain weight (`remove_weight_norm`). A
    /// no-op if already fused.
    pub fn fuse(&mut self) {
        if self.magnitude.is_some() {
            let w = self.materialize();
            self.weight = Param::new(w);
            self.magnitude = None;
        }
    }

    /// Re-impose weight normalisation on a plain weight (`g = ||w||`), e.g.
    /// to fine-tune from a fused checkpoint. A no-op if already normalised.
    pub fn normalize(&mut self) {
        if self.magnitude.is_none() {
            let g = Self::norm_of(&self.weight.value);
            self.magnitude = Some(Param::new(g));
        }
    }

    /// Cast the stored arrays to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.weight.value = self.weight.value.as_dtype(dtype);
        if let Some(g) = &mut self.magnitude {
            g.value = g.value.as_dtype(dtype);
        }
    }

    /// Adopt a checkpoint's weight, normalised (`v`, `g`) or plain, whichever
    /// the checkpoint holds.
    pub fn load(&mut self, direction_or_weight: Array, magnitude: Option<Array>) {
        self.weight = Param::new(direction_or_weight);
        self.magnitude = magnitude.map(Param::new);
    }
}

impl ModuleParameters for NormedWeight {
    fn num_parameters(&self) -> usize {
        1 + usize::from(self.magnitude.is_some())
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut out: ModuleParamRef<'_> = HashMap::new();
        match &self.magnitude {
            Some(g) => {
                out.insert(Rc::from("weight_v"), NestedValue::Value(&self.weight.value));
                out.insert(Rc::from("weight_g"), NestedValue::Value(&g.value));
            }
            None => {
                out.insert(Rc::from("weight"), NestedValue::Value(&self.weight.value));
            }
        }
        out
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut out: ModuleParamMut<'_> = HashMap::new();
        match &mut self.magnitude {
            Some(g) => {
                out.insert(
                    Rc::from("weight_v"),
                    NestedValue::Value(&mut self.weight.value),
                );
                out.insert(Rc::from("weight_g"), NestedValue::Value(&mut g.value));
            }
            None => {
                out.insert(
                    Rc::from("weight"),
                    NestedValue::Value(&mut self.weight.value),
                );
            }
        }
        out
    }
}

/// Implements `VisitLinears` as a no-op: the vocoder has no `Linear` layers
/// for an adapter to attach to, but nested parameter containers require the trait.
macro_rules! impl_no_linears {
    ($ty:ty) => {
        impl ::pmetal_runtime::compat::VisitLinears for $ty {
            fn visit_linears_mut(
                &mut self,
                _prefix: &str,
                _f: &mut dyn ::std::ops::FnMut(&str, &mut ::pmetal_runtime::compat::Linear),
            ) {
            }
        }
    };
}
pub(crate) use impl_no_linears;

/// Implements [`ModuleParameters`] for a layer with fields `weight: NormedWeight`
/// and `bias: Option<Array>`, flattening to `weight_v` / `weight_g` / `bias`.
macro_rules! impl_conv_params {
    ($ty:ty) => {
        impl_no_linears!($ty);

        impl ::pmetal_runtime::compat::ModuleParameters for $ty {
            fn num_parameters(&self) -> usize {
                self.weight.num_parameters() + usize::from(self.bias.is_some())
            }

            fn parameters(&self) -> ::pmetal_runtime::compat::ModuleParamRef<'_> {
                let mut out = self.weight.parameters();
                if let Some(b) = &self.bias {
                    out.insert(
                        ::std::rc::Rc::from("bias"),
                        ::pmetal_runtime::compat::NestedValue::Value(b),
                    );
                }
                out
            }

            fn parameters_mut(&mut self) -> ::pmetal_runtime::compat::ModuleParamMut<'_> {
                let mut out = self.weight.parameters_mut();
                if let Some(b) = &mut self.bias {
                    out.insert(
                        ::std::rc::Rc::from("bias"),
                        ::pmetal_runtime::compat::NestedValue::Value(b),
                    );
                }
                out
            }
        }
    };
}
pub(crate) use impl_conv_params;

/// A checkpoint's tensors by name, with bookkeeping for strict loading.
///
/// Layers `take` what they need; whatever is left over is a key the model has
/// no use for, which [`finish`](Self::finish) reports (the equivalent of
/// `strict=True`).
#[derive(Debug, Default)]
pub struct StateDict {
    tensors: HashMap<String, Array>,
    taken: HashSet<String>,
}

impl StateDict {
    /// A state dict over the given tensors.
    pub fn new(tensors: HashMap<String, Array>) -> Self {
        Self {
            tensors,
            taken: HashSet::new(),
        }
    }

    /// Whether `key` is present.
    pub fn contains(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// All tensor names, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Borrow a tensor without consuming it.
    pub fn get(&self, key: &str) -> Option<&Array> {
        self.tensors.get(key)
    }

    /// Take a tensor that must exist.
    pub fn take(&mut self, key: &str) -> Result<Array> {
        self.take_opt(key)
            .ok_or_else(|| VocoderError::WeightLoad(format!("checkpoint is missing `{key}`")))
    }

    /// Take a tensor if present.
    pub fn take_opt(&mut self, key: &str) -> Option<Array> {
        let t = self.tensors.get(key)?.clone();
        self.taken.insert(key.to_owned());
        Some(t)
    }

    /// Take a tensor and check its shape.
    pub fn take_shaped(&mut self, key: &str, shape: &[i32]) -> Result<Array> {
        let t = self.take(key)?;
        if t.shape() != shape {
            return Err(VocoderError::WeightLoad(format!(
                "`{key}` has shape {:?}, the model expects {shape:?}",
                t.shape()
            )));
        }
        Ok(t)
    }

    /// Fail if any tensor was never taken.
    pub fn finish(self) -> Result<()> {
        let mut unused: Vec<&String> = self
            .tensors
            .keys()
            .filter(|k| !self.taken.contains(*k))
            .collect();
        if unused.is_empty() {
            return Ok(());
        }
        unused.sort();
        let shown: Vec<&str> = unused.iter().take(8).map(|s| s.as_str()).collect();
        Err(VocoderError::WeightLoad(format!(
            "{} checkpoint tensor(s) match no model parameter, e.g. {}",
            unused.len(),
            shown.join(", ")
        )))
    }
}

/// `prefix.name`, or just `name` for an empty prefix.
pub fn join_key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_owned()
    } else {
        format!("{prefix}.{name}")
    }
}
