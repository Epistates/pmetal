//! Anti-aliased Multi-Periodicity (AMP) residual blocks.
//!
//! An AMP block is a residual stack whose nonlinearities are anti-aliased
//! Snake activations (BigVGAN, Lee et al. 2023). Two layouts exist upstream:
//!
//! * `AMPBlock1` - per dilation, `act -> conv(dilation d) -> act -> conv(dilation 1)`
//!   with a residual connection around each pair (the HiFi-GAN ResBlock1 shape);
//! * `AMPBlock2` - per dilation, `act -> conv(dilation d)` with a residual
//!   connection around each (ResBlock2).

use super::activation::{Activation1d, PeriodicActivation};
use super::conv::{Conv1d, Conv1dConfig};
use super::weight_norm::{StateDict, join_key};
use crate::config::{ActivationType, ResBlockKind};
use crate::error::{Result, VocoderError};
use pmetal_runtime::compat::{Array, Dtype};
use pmetal_runtime::impl_module_params;

/// One AMP residual block.
#[derive(Debug)]
pub struct AMPBlock {
    kind: ResBlockKind,
    /// Dilated convolutions, one per entry of the dilation list.
    pub convs1: Vec<Conv1d>,
    /// Dilation-1 convolutions that follow each of `convs1` (empty for
    /// [`ResBlockKind::Two`]).
    pub convs2: Vec<Conv1d>,
    /// Anti-aliased activations; two per dilation for kind one, one for kind two.
    pub activations: Vec<Activation1d>,
}

impl AMPBlock {
    /// Build a block of `channels` channels.
    ///
    /// * `kernel_size` - convolution kernel size.
    /// * `dilations` - dilation of each (first) convolution; its length is the
    ///   number of residual stages.
    /// * `activation` / `logscale` - the periodic activation and whether its
    ///   parameters are stored in log scale.
    pub fn new(
        kind: ResBlockKind,
        channels: i32,
        kernel_size: i32,
        dilations: &[i32],
        activation: ActivationType,
        logscale: bool,
    ) -> Result<Self> {
        if dilations.is_empty() {
            return Err(VocoderError::Config(
                "an AMP block needs at least one dilation".into(),
            ));
        }
        let conv = |dilation: i32| -> Result<Conv1d> {
            Conv1d::new(
                Conv1dConfig::new(channels, channels, kernel_size)
                    .dilation(dilation)
                    .same_padding(),
            )
        };
        let convs1 = dilations
            .iter()
            .map(|&d| conv(d))
            .collect::<Result<Vec<_>>>()?;
        let convs2 = match kind {
            ResBlockKind::One => dilations
                .iter()
                .map(|_| conv(1))
                .collect::<Result<Vec<_>>>()?,
            ResBlockKind::Two => Vec::new(),
        };
        let num_activations = convs1.len() + convs2.len();
        let activations = (0..num_activations)
            .map(|_| Activation1d::new(PeriodicActivation::new(activation, channels, logscale)?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            kind,
            convs1,
            convs2,
            activations,
        })
    }

    /// The block layout.
    pub fn kind(&self) -> ResBlockKind {
        self.kind
    }

    /// The checkpoint name of the dilated convolution list: `convs1` in
    /// `AMPBlock1` (paired with `convs2`), plain `convs` in `AMPBlock2`.
    fn dilated_name(&self) -> &'static str {
        match self.kind {
            ResBlockKind::One => "convs1",
            ResBlockKind::Two => "convs",
        }
    }

    /// Radius, in samples, of the block's receptive field (one side).
    pub fn receptive_radius(&self) -> i32 {
        let conv_radius = |c: &Conv1d| c.config().dilation * (c.config().kernel_size - 1) / 2;
        // Each Activation1d adds the reach of its resampling filters: the
        // upsampler reads `kernel / ratio` input samples around each output and
        // the decimator `kernel` upsampled ones (half as many at the input rate).
        let act_radius = self.activations.first().map_or(0, |a| a.receptive_radius());
        self.convs1.iter().map(conv_radius).sum::<i32>()
            + self.convs2.iter().map(conv_radius).sum::<i32>()
            + self.activations.len() as i32 * act_radius
    }

    /// Apply to `[batch, time, channels]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        match self.kind {
            ResBlockKind::One => {
                for ((c1, c2), acts) in self
                    .convs1
                    .iter()
                    .zip(&self.convs2)
                    .zip(self.activations.as_chunks::<2>().0)
                {
                    let mut xt = acts[0].forward(&x)?;
                    xt = c1.forward(&xt)?;
                    xt = acts[1].forward(&xt)?;
                    xt = c2.forward(&xt)?;
                    x = xt.add(&x);
                }
            }
            ResBlockKind::Two => {
                for (c, act) in self.convs1.iter().zip(&self.activations) {
                    let xt = c.forward(&act.forward(&x)?)?;
                    x = xt.add(&x);
                }
            }
        }
        Ok(x)
    }

    /// Fold weight normalisation into the convolution weights.
    pub fn fuse(&mut self) {
        self.convs1.iter_mut().for_each(Conv1d::fuse);
        self.convs2.iter_mut().for_each(Conv1d::fuse);
    }

    /// Switch every activation to the fused inference kernel (no gradient).
    pub fn prepare_inference(&mut self) {
        self.activations
            .iter_mut()
            .for_each(Activation1d::prepare_inference);
    }

    /// Cast parameters to `dtype`.
    pub fn cast(&mut self, dtype: Dtype) {
        self.convs1.iter_mut().for_each(|c| c.cast(dtype));
        self.convs2.iter_mut().for_each(|c| c.cast(dtype));
        self.activations.iter_mut().for_each(|a| a.cast(dtype));
    }

    /// Load from a PyTorch-layout state dict.
    pub fn load_state(&mut self, sd: &mut StateDict, prefix: &str) -> Result<()> {
        let dilated = self.dilated_name();
        for (i, c) in self.convs1.iter_mut().enumerate() {
            c.load_state(sd, &join_key(prefix, &format!("{dilated}.{i}")))?;
        }
        for (i, c) in self.convs2.iter_mut().enumerate() {
            c.load_state(sd, &join_key(prefix, &format!("convs2.{i}")))?;
        }
        for (i, a) in self.activations.iter_mut().enumerate() {
            a.load_state(sd, &join_key(prefix, &format!("activations.{i}")))?;
        }
        Ok(())
    }

    /// Append this block's tensors in PyTorch layout.
    pub fn save_state(&self, prefix: &str, out: &mut Vec<(String, Array)>) {
        let dilated = self.dilated_name();
        for (i, c) in self.convs1.iter().enumerate() {
            c.save_state(&join_key(prefix, &format!("{dilated}.{i}")), out);
        }
        for (i, c) in self.convs2.iter().enumerate() {
            c.save_state(&join_key(prefix, &format!("convs2.{i}")), out);
        }
        for (i, a) in self.activations.iter().enumerate() {
            a.save_state(&join_key(prefix, &format!("activations.{i}")), out);
        }
    }
}

impl_module_params!(AMPBlock; convs1, convs2, activations);
