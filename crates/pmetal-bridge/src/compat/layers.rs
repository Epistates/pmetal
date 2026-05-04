use super::{
    Array, Exception, ModuleParamMut, ModuleParamRef, ModuleParameters, Param, ops, random,
};
use std::collections::HashMap;
use std::rc::Rc;

fn fp8_weight_for_compute(weight: &Array) -> Array {
    if weight.dtype() == super::Dtype::Uint8 {
        ops::from_fp8(weight, super::Dtype::Bfloat16).expect("failed to dequantize FP8 weight")
    } else {
        weight.clone()
    }
}

fn linear_forward_array(x: &Array, weight: &Array, bias: Option<&Array>) -> Array {
    let weight = fp8_weight_for_compute(weight);
    let x = if weight.dtype() == super::Dtype::Bfloat16 && x.dtype() != super::Dtype::Bfloat16 {
        x.as_dtype(super::Dtype::Bfloat16.as_i32())
    } else {
        x.clone()
    };

    let output = x.matmul(&weight.t());
    if let Some(bias) = bias {
        output.add(bias)
    } else {
        output
    }
}

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
        linear_forward_array(x, &self.weight.value, self.bias.value.as_ref())
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
        let weight = fp8_weight_for_compute(&self.weight.value);
        x.rms_norm(Some(&weight), self.eps)
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
        let w = self.weight.value.as_ref().map(fp8_weight_for_compute);
        let b = self.bias.value.as_ref().map(fp8_weight_for_compute);
        x.layer_norm(w.as_ref(), b.as_ref(), self.eps)
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
            let ndim = x.ndim();
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
            let ndim = x.ndim();
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
            (Some(w), Some(b)) => {
                let w = fp8_weight_for_compute(w);
                let b = fp8_weight_for_compute(b);
                x.multiply(&w).add(&b)
            }
            (Some(w), None) => {
                let w = fp8_weight_for_compute(w);
                x.multiply(&w)
            }
            (None, Some(b)) => {
                let b = fp8_weight_for_compute(b);
                x.add(&b)
            }
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
        let weight = fp8_weight_for_compute(&self.weight.value);
        weight.take_axis(x, 0)
    }

    pub fn as_linear(&self, x: &Array) -> Array {
        linear_forward_array(x, &self.weight.value, None)
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

    #[allow(clippy::too_many_arguments)]
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
    layers: Vec<Box<dyn super::Module<&'static Array, Output = Array, Error = super::Exception>>>,
}

// Note: Sequential is intentionally left minimal. Full implementation would
// require boxing with `dyn Module` trait objects which require 'static lifetimes.
// For now it's a stub that satisfies type-checking.
impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({})", self.layers.len())
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
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
