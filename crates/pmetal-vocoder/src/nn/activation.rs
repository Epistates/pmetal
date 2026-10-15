//! Periodic activation functions for BigVGAN.
//!
//! BigVGAN uses Snake and SnakeBeta activations (Ziyin et al., 2020) that give
//! the generator a periodic inductive bias for modelling audio harmonics.
//! Activations here are channels-last, `[batch, time, channels]`, with one
//! learnable parameter per channel.

use super::resample::{DownSample1d, UpSample1d};
use super::weight_norm::{StateDict, impl_no_linears, join_key};
use crate::config::ActivationType;
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{
    Array, Dtype, ModuleParamMut, ModuleParamRef, ModuleParameters, Param,
};
use pmetal_runtime::impl_module_params;

/// Added to the divisor of the periodic term, as in the reference
/// implementation, so a parameter that reaches zero cannot produce `inf`.
const NO_DIV_BY_ZERO: f32 = 1e-9;

/// `exp(p)` when the parameter is stored in log scale, else `p`.
fn effective(param: &Array, logscale: bool) -> Array {
    if logscale { param.exp() } else { param.clone() }
}

/// `x + sin^2(alpha x) / (denominator + eps)`, computed in float32 and
/// returned in the input's dtype (`sin` of a large phase is not safe in half
/// precision).
fn snake_core(x: &Array, alpha: &Array, denominator: &Array) -> Array {
    let dtype = x.dtype();
    let x32 = x.as_dtype(Dtype::Float32);
    let sin = x32.multiply(&alpha.as_dtype(Dtype::Float32)).sin();
    let eps = Array::from_f32(NO_DIV_BY_ZERO);
    let periodic = sin
        .multiply(&sin)
        .divide(&denominator.as_dtype(Dtype::Float32).add(&eps));
    let y = x32.add(&periodic);
    if dtype == Dtype::Float32 {
        y
    } else {
        y.as_dtype(dtype)
    }
}

fn param_vector(channels: i32, logscale: bool) -> Result<Param<Array>> {
    if channels < 1 {
        return Err(VocoderError::Config(format!(
            "activation needs at least one channel, got {channels}"
        )));
    }
    // Log-scale parameters start at 0 (exp(0) = 1), linear ones at 1.
    let init = if logscale { 0.0 } else { 1.0 };
    Ok(Param::new(Array::full(&[channels], init, Dtype::Float32)))
}

/// Snake activation, `x + sin^2(alpha x) / alpha`.
///
/// The learnable per-channel `alpha` sets the frequency of the periodic
/// component; larger values capture higher-frequency content.
#[derive(Debug)]
pub struct Snake {
    /// Per-channel frequency parameter `[channels]`.
    pub alpha: Param<Array>,
    /// Whether `alpha` is stored in log scale (`exp(alpha)` is used).
    pub alpha_logscale: bool,
}

impl Snake {
    /// A Snake activation over `channels` channels, initialised to `alpha = 1`.
    pub fn new(channels: i32, alpha_logscale: bool) -> Result<Self> {
        Ok(Self {
            alpha: param_vector(channels, alpha_logscale)?,
            alpha_logscale,
        })
    }

    /// Number of channels.
    pub fn channels(&self) -> i32 {
        self.alpha.value.dim(0)
    }

    /// Apply to `[batch, time, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let alpha = effective(&self.alpha.value, self.alpha_logscale);
        Ok(snake_core(x, &alpha, &alpha))
    }
}

impl_module_params!(Snake; alpha);

/// SnakeBeta activation, `x + sin^2(alpha x) / beta`.
///
/// Separate per-channel `alpha` (frequency) and `beta` (magnitude) make the
/// periodic term's amplitude independent of its frequency.
#[derive(Debug)]
pub struct SnakeBeta {
    /// Per-channel frequency parameter `[channels]`.
    pub alpha: Param<Array>,
    /// Per-channel magnitude parameter `[channels]`.
    pub beta: Param<Array>,
    /// Whether both parameters are stored in log scale.
    pub logscale: bool,
}

impl SnakeBeta {
    /// A SnakeBeta activation over `channels` channels, initialised to
    /// `alpha = beta = 1`.
    pub fn new(channels: i32, logscale: bool) -> Result<Self> {
        Ok(Self {
            alpha: param_vector(channels, logscale)?,
            beta: param_vector(channels, logscale)?,
            logscale,
        })
    }

    /// Number of channels.
    pub fn channels(&self) -> i32 {
        self.alpha.value.dim(0)
    }

    /// Apply to `[batch, time, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let alpha = effective(&self.alpha.value, self.logscale);
        let beta = effective(&self.beta.value, self.logscale);
        Ok(snake_core(x, &alpha, &beta))
    }
}

impl_module_params!(SnakeBeta; alpha, beta);

/// Either periodic activation, chosen by [`ActivationType`].
#[derive(Debug)]
pub enum PeriodicActivation {
    /// [`Snake`].
    Snake(Snake),
    /// [`SnakeBeta`].
    SnakeBeta(SnakeBeta),
}

impl PeriodicActivation {
    /// Build the activation `kind` over `channels` channels.
    pub fn new(kind: ActivationType, channels: i32, logscale: bool) -> Result<Self> {
        Ok(match kind {
            ActivationType::Snake => Self::Snake(Snake::new(channels, logscale)?),
            ActivationType::SnakeBeta => Self::SnakeBeta(SnakeBeta::new(channels, logscale)?),
        })
    }

    /// Number of channels.
    pub fn channels(&self) -> i32 {
        match self {
            Self::Snake(a) => a.channels(),
            Self::SnakeBeta(a) => a.channels(),
        }
    }

    /// Apply to `[batch, time, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Snake(a) => a.forward(x),
            Self::SnakeBeta(a) => a.forward(x),
        }
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        let params: Vec<&mut Param<Array>> = match self {
            Self::Snake(a) => vec![&mut a.alpha],
            Self::SnakeBeta(a) => vec![&mut a.alpha, &mut a.beta],
        };
        for p in params {
            p.value = p.value.as_dtype(dtype);
        }
    }

    /// Load `prefix.alpha` (and `prefix.beta`) from a state dict.
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let channels = self.channels();
        match self {
            Self::Snake(a) => {
                a.alpha.value = sd.take_shaped(&join_key(prefix, "alpha"), &[channels])?;
            }
            Self::SnakeBeta(a) => {
                a.alpha.value = sd.take_shaped(&join_key(prefix, "alpha"), &[channels])?;
                a.beta.value = sd.take_shaped(&join_key(prefix, "beta"), &[channels])?;
            }
        }
        Ok(())
    }

    /// Append `prefix.alpha` (and `prefix.beta`).
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        match self {
            Self::Snake(a) => out.push((join_key(prefix, "alpha"), a.alpha.value.clone())),
            Self::SnakeBeta(a) => {
                out.push((join_key(prefix, "alpha"), a.alpha.value.clone()));
                out.push((join_key(prefix, "beta"), a.beta.value.clone()));
            }
        }
    }
}

impl_no_linears!(PeriodicActivation);

impl ModuleParameters for PeriodicActivation {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Snake(a) => a.num_parameters(),
            Self::SnakeBeta(a) => a.num_parameters(),
        }
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        match self {
            Self::Snake(a) => a.parameters(),
            Self::SnakeBeta(a) => a.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        match self {
            Self::Snake(a) => a.parameters_mut(),
            Self::SnakeBeta(a) => a.parameters_mut(),
        }
    }
}

/// An anti-aliased activation: upsample, activate, low-pass and downsample.
///
/// The nonlinearity creates harmonics above the original Nyquist frequency;
/// evaluating it at `up_ratio` times the sample rate and low-pass filtering
/// before decimating stops those harmonics folding back as aliasing.
#[derive(Debug)]
pub struct Activation1d {
    /// The periodic nonlinearity, applied at the upsampled rate.
    pub act: PeriodicActivation,
    /// Interpolating upsampler.
    pub upsample: UpSample1d,
    /// Low-pass decimator.
    pub downsample: DownSample1d,
}

impl Activation1d {
    /// The reference's defaults: 2x up and down, 12-tap filters.
    pub fn new(act: PeriodicActivation) -> Result<Self> {
        Self::with_resampling(act, 2, 2, 12, 12)
    }

    /// Explicit resampling ratios and filter sizes.
    pub fn with_resampling(
        act: PeriodicActivation,
        up_ratio: i32,
        down_ratio: i32,
        up_kernel_size: i32,
        down_kernel_size: i32,
    ) -> Result<Self> {
        let channels = act.channels();
        Ok(Self {
            upsample: UpSample1d::new(up_ratio, up_kernel_size, channels)?,
            downsample: DownSample1d::new(down_ratio, down_kernel_size, channels)?,
            act,
        })
    }

    /// Apply to `[batch, time, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let up = self.upsample.forward(x)?;
        let activated = self.act.forward(&up)?;
        self.downsample.forward(&activated)
    }

    /// How many input samples on each side an output sample can depend on:
    /// the upsampler's reach plus the decimator's, in input samples.
    pub fn receptive_radius(&self) -> i32 {
        self.upsample.input_reach() + self.downsample.input_reach(self.upsample.ratio())
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.act.cast(dtype);
    }

    /// Load the activation parameters and, when present, the filter buffers.
    ///
    /// The reference stores the resampling filters as buffers; they are a pure
    /// function of the ratio and kernel size, so a checkpoint that lacks them
    /// is fine, and one that has them must agree on the length.
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        self.act.load_state(sd, &join_key(prefix, "act"))?;
        let up_key = join_key(prefix, "upsample.filter");
        if let Some(f) = sd.take_opt(&up_key) {
            self.upsample.set_taps(filter_taps(&f, &up_key)?)?;
        }
        let down_key = join_key(prefix, "downsample.lowpass.filter");
        if let Some(f) = sd.take_opt(&down_key) {
            self.downsample.set_taps(filter_taps(&f, &down_key)?)?;
        }
        Ok(())
    }

    /// Append the activation parameters and filter buffers.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        self.act.save_state(&join_key(prefix, "act"), out);
        let buffer = |taps: &[f32]| Array::from_f32_slice(taps, &[1, 1, taps.len() as i32]);
        out.push((
            join_key(prefix, "upsample.filter"),
            buffer(self.upsample.taps()),
        ));
        out.push((
            join_key(prefix, "downsample.lowpass.filter"),
            buffer(self.downsample.taps()),
        ));
    }
}

/// The taps of a `[1, 1, taps]` filter buffer.
fn filter_taps(buffer: &Array, key: &str) -> Result<Vec<f32>> {
    if buffer.ndim() != 3 || buffer.dim(0) != 1 || buffer.dim(1) != 1 {
        return Err(VocoderError::WeightLoad(format!(
            "`{key}` has shape {:?}, expected [1, 1, taps]",
            buffer.shape()
        )));
    }
    Ok(buffer.as_dtype(Dtype::Float32).as_slice::<f32>().to_vec())
}

impl_module_params!(Activation1d; act);
