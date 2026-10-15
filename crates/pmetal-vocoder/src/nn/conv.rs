//! Convolution layers for BigVGAN and its discriminators.
//!
//! Layers are channels-last (MLX's native layout): `[batch, time, channels]`
//! for 1-D and `[batch, height, width, channels]` for 2-D, with weights as
//! `[out, kernel.., in]`. Checkpoints use PyTorch's channels-first layout; the
//! `load_state` / `save_state` methods convert at the boundary.
//!
//! All layers can run under weight normalisation (training) or fused to a
//! plain weight (inference); see [`NormedWeight`].

use super::overlap::overlap_add;
use super::weight_norm::{NormedWeight, StateDict, impl_conv_params, impl_no_linears, join_key};
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, Dtype, ops, random};

/// PyTorch's default convolution initialisation: weight and bias drawn from
/// `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`.
fn uniform_init(shape: &[i32], fan_in: i32) -> Array {
    let bound = (1.0 / fan_in.max(1) as f32).sqrt();
    let u = random::uniform(shape, Dtype::Float32);
    u.multiply(&Array::from_f32(2.0 * bound))
        .add(&Array::from_f32(-bound))
}

fn make_bias(use_bias: bool, out_channels: i32, fan_in: i32) -> Option<Array> {
    use_bias.then(|| uniform_init(&[out_channels], fan_in))
}

fn positive(what: &str, value: i32) -> Result<()> {
    if value < 1 {
        return Err(VocoderError::Config(format!(
            "{what} must be >= 1, got {value}"
        )));
    }
    Ok(())
}

/// Where a layer's tensors sit in a PyTorch checkpoint.
struct TorchLayout {
    /// Shape of `weight` / `weight_v`.
    weight_shape: Vec<i32>,
    /// Shape of `weight_g`.
    g_shape: Vec<i32>,
    /// Axis permutation from the checkpoint layout to ours.
    to_ours: &'static [i32],
    /// The inverse permutation.
    to_torch: &'static [i32],
}

/// A layer's weight and bias, as the checkpoint code sees them.
struct ConvParts<'a> {
    weight: &'a mut NormedWeight,
    bias: &'a mut Option<Array>,
    out_channels: i32,
}

impl ConvParts<'_> {
    /// Load `prefix.{weight_v, weight_g | weight}` and `prefix.bias`.
    fn load(&mut self, layout: &TorchLayout, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let v_key = join_key(prefix, "weight_v");
        if sd.contains(&v_key) {
            let v = sd.take_shaped(&v_key, &layout.weight_shape)?;
            let g = sd.take_shaped(&join_key(prefix, "weight_g"), &layout.g_shape)?;
            self.weight.load(v.transpose_axes(layout.to_ours), Some(g));
        } else {
            let w = sd.take_shaped(&join_key(prefix, "weight"), &layout.weight_shape)?;
            self.weight.load(w.transpose_axes(layout.to_ours), None);
        }
        let bias_key = join_key(prefix, "bias");
        match self.bias {
            Some(b) => *b = sd.take_shaped(&bias_key, &[self.out_channels])?,
            None => {
                if sd.contains(&bias_key) {
                    return Err(VocoderError::WeightLoad(format!(
                        "checkpoint has `{bias_key}` but the layer was built without a bias"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Write a layer's tensors under `prefix` in the reference naming and layout.
fn save_conv_state(
    weight: &NormedWeight,
    bias: &Option<Array>,
    layout: &TorchLayout,
    prefix: &str,
    out: &mut Vec<(String, Array)>,
) {
    let w = weight.weight.value.transpose_axes(layout.to_torch);
    if let Some(g) = &weight.magnitude {
        out.push((join_key(prefix, "weight_v"), w));
        out.push((join_key(prefix, "weight_g"), g.value.clone()));
    } else {
        out.push((join_key(prefix, "weight"), w));
    }
    if let Some(b) = bias {
        out.push((join_key(prefix, "bias"), b.clone()));
    }
}

/// Settings for a [`Conv1d`].
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    /// Input channels.
    pub in_channels: i32,
    /// Output channels.
    pub out_channels: i32,
    /// Kernel size.
    pub kernel_size: i32,
    /// Stride.
    pub stride: i32,
    /// Zero padding added to both ends.
    pub padding: i32,
    /// Dilation.
    pub dilation: i32,
    /// Channel groups.
    pub groups: i32,
    /// Whether the layer has a bias.
    pub bias: bool,
    /// Whether the weight is weight-normalised.
    pub weight_norm: bool,
}

impl Conv1dConfig {
    /// A stride-1, undilated, weight-normalised convolution with bias.
    pub fn new(in_channels: i32, out_channels: i32, kernel_size: i32) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            weight_norm: true,
        }
    }

    /// Set the stride.
    pub fn stride(mut self, stride: i32) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn padding(mut self, padding: i32) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation.
    pub fn dilation(mut self, dilation: i32) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of channel groups.
    pub fn groups(mut self, groups: i32) -> Self {
        self.groups = groups;
        self
    }

    /// Choose whether the layer has a bias.
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Choose whether the weight is weight-normalised.
    pub fn weight_norm(mut self, weight_norm: bool) -> Self {
        self.weight_norm = weight_norm;
        self
    }

    /// The "same" padding `dilation * (kernel - 1) / 2` the reference's
    /// `get_padding` computes (exact for odd kernels).
    pub fn same_padding(mut self) -> Self {
        self.padding = (self.kernel_size * self.dilation - self.dilation) / 2;
        self
    }
}

/// A 1-D convolution on `[batch, time, channels]`.
#[derive(Debug)]
pub struct Conv1d {
    /// Weight `[out, kernel, in / groups]`, normalised or plain.
    pub weight: NormedWeight,
    /// Optional bias `[out]`.
    pub bias: Option<Array>,
    config: Conv1dConfig,
}

impl Conv1d {
    /// A freshly initialised layer (PyTorch's default init).
    pub fn new(config: Conv1dConfig) -> Result<Self> {
        positive("in_channels", config.in_channels)?;
        positive("out_channels", config.out_channels)?;
        positive("kernel_size", config.kernel_size)?;
        positive("stride", config.stride)?;
        positive("dilation", config.dilation)?;
        positive("groups", config.groups)?;
        if config.in_channels % config.groups != 0 || config.out_channels % config.groups != 0 {
            return Err(VocoderError::Config(format!(
                "channels ({}, {}) must be divisible by groups ({})",
                config.in_channels, config.out_channels, config.groups
            )));
        }
        let per_group = config.in_channels / config.groups;
        let fan_in = per_group * config.kernel_size;
        let weight = uniform_init(
            &[config.out_channels, config.kernel_size, per_group],
            fan_in,
        );
        Ok(Self {
            weight: NormedWeight::new(weight, config.weight_norm),
            bias: make_bias(config.bias, config.out_channels, fan_in),
            config,
        })
    }

    /// The layer's settings.
    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    /// Output length for an input of `length` samples.
    pub fn output_len(&self, length: i32) -> i32 {
        let c = &self.config;
        (length + 2 * c.padding - c.dilation * (c.kernel_size - 1) - 1) / c.stride + 1
    }

    /// Apply to `[batch, time, in_channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let c = &self.config;
        if x.ndim() != 3 || x.dim(2) != c.in_channels {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1, c.in_channels],
                actual: x.shape().to_vec(),
            });
        }
        let weight = self.weight.materialize().as_dtype(x.dtype());
        let y = ops::conv1d(x, &weight, c.stride, c.padding, c.dilation, c.groups);
        Ok(match &self.bias {
            Some(b) => y.add(&b.as_dtype(x.dtype())),
            None => y,
        })
    }

    /// Fold weight normalisation into the weight (inference).
    pub fn fuse(&mut self) {
        self.weight.fuse();
    }

    /// Re-impose weight normalisation on a fused weight (to train from one).
    pub fn normalize(&mut self) {
        self.weight.normalize();
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.weight.cast(dtype);
        if let Some(b) = &mut self.bias {
            *b = b.as_dtype(dtype);
        }
    }

    /// Load from a PyTorch-layout state dict (`[out, in, kernel]` weights).
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let layout = self.torch_layout();
        ConvParts {
            weight: &mut self.weight,
            bias: &mut self.bias,
            out_channels: self.config.out_channels,
        }
        .load(&layout, sd, prefix)
    }

    /// Append this layer's tensors in PyTorch layout.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        save_conv_state(&self.weight, &self.bias, &self.torch_layout(), prefix, out);
    }

    fn torch_layout(&self) -> TorchLayout {
        let c = &self.config;
        TorchLayout {
            weight_shape: vec![c.out_channels, c.in_channels / c.groups, c.kernel_size],
            g_shape: vec![c.out_channels, 1, 1],
            to_ours: &[0, 2, 1],
            to_torch: &[0, 2, 1],
        }
    }
}

impl_conv_params!(Conv1d);

/// A transposed 1-D convolution on `[batch, time, channels]`, for upsampling.
///
/// Implemented as one matmul followed by an overlap-add of the `kernel / stride`
/// shifted blocks, which keeps the whole layer on the matrix units instead of
/// running a zero-stuffed convolution.
#[derive(Debug)]
pub struct ConvTranspose1d {
    /// Weight `[in, kernel, out]`, normalised or plain.
    pub weight: NormedWeight,
    /// Optional bias `[out]`.
    pub bias: Option<Array>,
    in_channels: i32,
    out_channels: i32,
    kernel_size: i32,
    stride: i32,
    padding: i32,
}

impl ConvTranspose1d {
    /// A freshly initialised layer (PyTorch's default init), weight-normalised.
    ///
    /// The output has `(length - 1) * stride - 2 * padding + kernel_size`
    /// samples; `padding = (kernel_size - stride) / 2` gives exactly
    /// `length * stride`.
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: i32,
        padding: i32,
        bias: bool,
    ) -> Result<Self> {
        positive("in_channels", in_channels)?;
        positive("out_channels", out_channels)?;
        positive("kernel_size", kernel_size)?;
        positive("stride", stride)?;
        if padding < 0 {
            return Err(VocoderError::Config(format!(
                "padding must be >= 0, got {padding}"
            )));
        }
        // PyTorch's fan-in for a transposed weight [in, out, k] is out * k.
        let fan_in = out_channels * kernel_size;
        let weight = uniform_init(&[in_channels, kernel_size, out_channels], fan_in);
        Ok(Self {
            weight: NormedWeight::new(weight, true),
            bias: make_bias(bias, out_channels, fan_in),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Output length for an input of `length` samples.
    pub fn output_len(&self, length: i32) -> i32 {
        (length - 1) * self.stride - 2 * self.padding + self.kernel_size
    }

    /// `(stride, kernel_size, padding)`.
    pub fn geometry(&self) -> (i32, i32, i32) {
        (self.stride, self.kernel_size, self.padding)
    }

    /// Apply to `[batch, time, in_channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if x.ndim() != 3 || x.dim(2) != self.in_channels {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1, self.in_channels],
                actual: x.shape().to_vec(),
            });
        }
        let (batch, length) = (x.dim(0), x.dim(1));
        let (k, s, o) = (self.kernel_size, self.stride, self.out_channels);
        let out_len = self.output_len(length);
        if out_len < 1 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, 1, self.in_channels],
                actual: x.shape().to_vec(),
            });
        }

        // Every input sample scatters a kernel-long, out-channel-wide patch.
        let weight = self
            .weight
            .materialize()
            .as_dtype(x.dtype())
            .reshape(&[self.in_channels, k * o]);
        let patches = x
            .reshape(&[batch * length, self.in_channels])
            .matmul(&weight);

        let full = overlap_add(&patches.reshape(&[batch, length, k, o]), s);
        let y = ops::slice_axis(&full, 1, self.padding, self.padding + out_len);
        Ok(match &self.bias {
            Some(b) => y.add(&b.as_dtype(x.dtype())),
            None => y,
        })
    }

    /// Fold weight normalisation into the weight (inference).
    pub fn fuse(&mut self) {
        self.weight.fuse();
    }

    /// Re-impose weight normalisation on a fused weight (to train from one).
    pub fn normalize(&mut self) {
        self.weight.normalize();
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.weight.cast(dtype);
        if let Some(b) = &mut self.bias {
            *b = b.as_dtype(dtype);
        }
    }

    /// Load from a PyTorch-layout state dict (`[in, out, kernel]` weights).
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let layout = self.torch_layout();
        ConvParts {
            weight: &mut self.weight,
            bias: &mut self.bias,
            out_channels: self.out_channels,
        }
        .load(&layout, sd, prefix)
    }

    /// Append this layer's tensors in PyTorch layout.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        save_conv_state(&self.weight, &self.bias, &self.torch_layout(), prefix, out);
    }

    fn torch_layout(&self) -> TorchLayout {
        TorchLayout {
            weight_shape: vec![self.in_channels, self.out_channels, self.kernel_size],
            g_shape: vec![self.in_channels, 1, 1],
            to_ours: &[0, 2, 1],
            to_torch: &[0, 2, 1],
        }
    }
}

impl_conv_params!(ConvTranspose1d);

/// Settings for a [`Conv2d`]. Pairs are `(height, width)`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    /// Input channels.
    pub in_channels: i32,
    /// Output channels.
    pub out_channels: i32,
    /// Kernel size.
    pub kernel_size: (i32, i32),
    /// Stride.
    pub stride: (i32, i32),
    /// Zero padding.
    pub padding: (i32, i32),
    /// Dilation.
    pub dilation: (i32, i32),
    /// Whether the weight is weight-normalised.
    pub weight_norm: bool,
}

impl Conv2dConfig {
    /// A stride-1, undilated, unpadded, weight-normalised convolution.
    pub fn new(in_channels: i32, out_channels: i32, kernel_size: (i32, i32)) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            weight_norm: true,
        }
    }

    /// Set the stride.
    pub fn stride(mut self, stride: (i32, i32)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn padding(mut self, padding: (i32, i32)) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation.
    pub fn dilation(mut self, dilation: (i32, i32)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Choose whether the weight is weight-normalised.
    pub fn weight_norm(mut self, weight_norm: bool) -> Self {
        self.weight_norm = weight_norm;
        self
    }

    /// The reference's `get_padding`: `dilation * (kernel - 1) / 2` per axis.
    pub fn same_padding(mut self) -> Self {
        self.padding = (
            (self.kernel_size.0 - 1) * self.dilation.0 / 2,
            (self.kernel_size.1 - 1) * self.dilation.1 / 2,
        );
        self
    }
}

/// A 2-D convolution on `[batch, height, width, channels]`.
#[derive(Debug)]
pub struct Conv2d {
    /// Weight `[out, kh, kw, in]`, normalised or plain.
    pub weight: NormedWeight,
    /// Bias `[out]`.
    pub bias: Option<Array>,
    config: Conv2dConfig,
}

impl Conv2d {
    /// A freshly initialised layer (PyTorch's default init, with bias).
    pub fn new(config: Conv2dConfig) -> Result<Self> {
        positive("in_channels", config.in_channels)?;
        positive("out_channels", config.out_channels)?;
        for (name, (a, b)) in [
            ("kernel_size", config.kernel_size),
            ("stride", config.stride),
            ("dilation", config.dilation),
        ] {
            positive(name, a)?;
            positive(name, b)?;
        }
        let (kh, kw) = config.kernel_size;
        let fan_in = config.in_channels * kh * kw;
        let weight = uniform_init(&[config.out_channels, kh, kw, config.in_channels], fan_in);
        Ok(Self {
            weight: NormedWeight::new(weight, config.weight_norm),
            bias: make_bias(true, config.out_channels, fan_in),
            config,
        })
    }

    /// The layer's settings.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    /// Apply to `[batch, height, width, in_channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let c = &self.config;
        if x.ndim() != 4 || x.dim(3) != c.in_channels {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1, -1, c.in_channels],
                actual: x.shape().to_vec(),
            });
        }
        let weight = self.weight.materialize().as_dtype(x.dtype());
        let y = x.conv2d(
            &weight,
            c.stride.0,
            c.stride.1,
            c.padding.0,
            c.padding.1,
            c.dilation.0,
            c.dilation.1,
            1,
        );
        Ok(match &self.bias {
            Some(b) => y.add(&b.as_dtype(x.dtype())),
            None => y,
        })
    }

    /// Fold weight normalisation into the weight (inference).
    pub fn fuse(&mut self) {
        self.weight.fuse();
    }

    /// Re-impose weight normalisation on a fused weight (to train from one).
    pub fn normalize(&mut self) {
        self.weight.normalize();
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.weight.cast(dtype);
        if let Some(b) = &mut self.bias {
            *b = b.as_dtype(dtype);
        }
    }

    /// Load from a PyTorch-layout state dict (`[out, in, kh, kw]` weights).
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let layout = self.torch_layout();
        ConvParts {
            weight: &mut self.weight,
            bias: &mut self.bias,
            out_channels: self.config.out_channels,
        }
        .load(&layout, sd, prefix)
    }

    /// Append this layer's tensors in PyTorch layout.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        save_conv_state(&self.weight, &self.bias, &self.torch_layout(), prefix, out);
    }

    fn torch_layout(&self) -> TorchLayout {
        let c = &self.config;
        TorchLayout {
            weight_shape: vec![
                c.out_channels,
                c.in_channels,
                c.kernel_size.0,
                c.kernel_size.1,
            ],
            g_shape: vec![c.out_channels, 1, 1, 1],
            to_ours: &[0, 2, 3, 1],
            to_torch: &[0, 3, 1, 2],
        }
    }
}

impl_conv_params!(Conv2d);
