//! BigVGAN generator for mel-to-waveform synthesis.
//!
//! The generator converts mel spectrograms to waveforms with a stack of
//! transposed-convolution upsamplers, each followed by a bank of parallel AMP
//! (anti-aliased multi-periodicity) residual blocks whose outputs are averaged
//! (BigVGAN, Lee et al. 2023). Checkpoints released with the reference
//! implementation load as they are.
//!
//! Inside the model tensors are channels-last; [`BigVGAN::forward`] takes and
//! returns the reference's channels-first layout.

mod load;
mod streaming;

pub use streaming::StreamingSynthesizer;

use crate::config::{ActivationType, BigVGANConfig};
use crate::error::{Result, VocoderError};
use crate::nn::{
    AMPBlock, Activation1d, Conv1d, Conv1dConfig, ConvTranspose1d, PeriodicActivation, StateDict,
    join_key,
};
use pmetal_runtime::compat::{Array, Dtype, ops};
use pmetal_runtime::impl_module_params;

/// BigVGAN neural vocoder.
///
/// Converts mel spectrograms `[batch, n_mels, frames]` to waveforms
/// `[batch, 1, frames * hop_size]`.
#[derive(Debug)]
pub struct BigVGAN {
    /// Model configuration.
    pub config: BigVGANConfig,
    /// Input convolution from mel bins to the widest hidden size.
    pub conv_pre: Conv1d,
    /// Transposed-convolution upsamplers, one per stage.
    pub ups: Vec<ConvTranspose1d>,
    /// AMP blocks, `num_kernels` per stage, stage-major
    /// (`resblocks[stage * num_kernels + j]`).
    pub resblocks: Vec<AMPBlock>,
    /// Anti-aliased activation before the output convolution.
    pub activation_post: Activation1d,
    /// Output convolution to a single channel.
    pub conv_post: Conv1d,
}

impl BigVGAN {
    /// A freshly initialised generator.
    pub fn new(config: BigVGANConfig) -> Result<Self> {
        config.validate()?;
        let channels = config.channel_sizes();
        let stages = config.num_upsample_stages();

        let conv_pre = Conv1d::new(
            Conv1dConfig::new(config.num_mels, config.upsample_initial_channel, 7).padding(3),
        )?;

        let mut ups = Vec::with_capacity(stages);
        let mut resblocks = Vec::with_capacity(stages * config.resblock_kernel_sizes.len());
        for i in 0..stages {
            let (rate, kernel) = (config.upsample_rates[i], config.upsample_kernel_sizes[i]);
            ups.push(ConvTranspose1d::new(
                channels[i],
                channels[i + 1],
                kernel,
                rate,
                (kernel - rate) / 2,
                true,
            )?);
            for (&k, dilations) in config
                .resblock_kernel_sizes
                .iter()
                .zip(&config.resblock_dilation_sizes)
            {
                resblocks.push(AMPBlock::new(
                    config.resblock,
                    channels[i + 1],
                    k,
                    dilations,
                    config.activation,
                    config.snake_logscale,
                )?);
            }
        }

        let final_channels = channels[stages];
        let activation_post = Activation1d::new(PeriodicActivation::new(
            config.activation,
            final_channels,
            config.snake_logscale,
        )?)?;
        let conv_post = Conv1d::new(
            Conv1dConfig::new(final_channels, 1, 7)
                .padding(3)
                .bias(config.use_bias_at_final),
        )?;

        Ok(Self {
            config,
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
        })
    }

    /// Number of parallel AMP blocks per stage.
    pub fn num_kernels(&self) -> usize {
        self.config.resblock_kernel_sizes.len()
    }

    /// Samples produced per mel frame.
    pub fn hop_size(&self) -> i32 {
        self.config.hop_size
    }

    /// Synthesise from a channels-last mel `[batch, frames, n_mels]` to a
    /// channels-last waveform `[batch, frames * hop_size, 1]`.
    pub fn forward_nlc(&self, mel: &Array) -> Result<Array> {
        if mel.ndim() != 3 || mel.dim(2) != self.config.num_mels {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, -1, self.config.num_mels],
                actual: mel.shape().to_vec(),
            });
        }
        let mut x = self.conv_pre.forward(mel)?;
        let num_kernels = self.num_kernels();
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x)?;
            let mut sum: Option<Array> = None;
            for block in &self.resblocks[i * num_kernels..(i + 1) * num_kernels] {
                let y = block.forward(&x)?;
                sum = Some(match sum {
                    Some(s) => s.add(&y),
                    None => y,
                });
            }
            x = sum
                .expect("validated: at least one resblock per stage")
                .divide(&Array::from_f32(num_kernels as f32).as_dtype(x.dtype()));
        }
        x = self.activation_post.forward(&x)?;
        x = self.conv_post.forward(&x)?;
        Ok(if self.config.use_tanh_at_final {
            ops::tanh(&x)
        } else {
            let lo = Array::from_f32(-1.0).as_dtype(x.dtype());
            let hi = Array::from_f32(1.0).as_dtype(x.dtype());
            ops::clip(&x, Some(&lo), Some(&hi))
        })
    }

    /// Synthesise from a mel spectrogram `[batch, n_mels, frames]` to a waveform
    /// `[batch, 1, frames * hop_size]` (the reference's layout).
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        if mel.ndim() != 3 {
            return Err(VocoderError::ShapeMismatch {
                expected: vec![-1, self.config.num_mels, -1],
                actual: mel.shape().to_vec(),
            });
        }
        let audio = self.forward_nlc(&mel.transpose_axes(&[0, 2, 1]))?;
        Ok(audio.transpose_axes(&[0, 2, 1]))
    }

    /// Synthesise audio from `[n_mels, frames]` or `[batch, n_mels, frames]`,
    /// returning `[samples]` or `[batch, samples]`.
    pub fn generate(&self, mel: &Array) -> Result<Array> {
        let unbatched = mel.ndim() == 2;
        let batched = if unbatched {
            mel.reshape(&[1, mel.dim(0), mel.dim(1)])
        } else {
            mel.clone()
        };
        let audio = self.forward(&batched)?;
        let audio = audio.squeeze_axes(&[1]);
        Ok(if unbatched {
            audio.squeeze_axes(&[0])
        } else {
            audio
        })
    }

    /// Fold weight normalisation into the convolution weights, as the
    /// reference does before inference. Training is not possible afterwards
    /// without re-imposing the normalisation.
    pub fn fuse_weight_norm(&mut self) {
        self.conv_pre.fuse();
        self.ups.iter_mut().for_each(ConvTranspose1d::fuse);
        self.resblocks.iter_mut().for_each(AMPBlock::fuse);
        self.conv_post.fuse();
    }

    /// Get the model ready for fast synthesis: fold weight normalisation into
    /// the weights ([`fuse_weight_norm`](Self::fuse_weight_norm)) and run every
    /// anti-aliased activation as one fused GPU kernel. Neither can be
    /// differentiated, so this is for inference; call it after the final
    /// [`to_dtype`](Self::to_dtype) and weight loading (either drops the fused
    /// kernels again).
    pub fn prepare_for_inference(&mut self) {
        self.fuse_weight_norm();
        self.resblocks
            .iter_mut()
            .for_each(AMPBlock::prepare_inference);
        self.activation_post.prepare_inference();
    }

    /// Undo [`prepare_for_inference`](Self::prepare_for_inference) so the model
    /// can be trained: the activations return to differentiable ops and
    /// weight normalisation is re-imposed on any fused convolution (`g` starts
    /// at the weight's norm, so the function is unchanged).
    pub fn prepare_for_training(&mut self) {
        self.conv_pre.normalize();
        self.ups.iter_mut().for_each(ConvTranspose1d::normalize);
        self.resblocks.iter_mut().for_each(|b| {
            b.normalize();
            b.unprepare_inference();
        });
        self.activation_post.unprepare();
        self.conv_post.normalize();
    }

    /// Cast every parameter to `dtype`. The periodic activations always
    /// compute in float32 internally; convolutions run in `dtype`.
    pub fn to_dtype(&mut self, dtype: Dtype) {
        self.conv_pre.cast(dtype);
        self.ups.iter_mut().for_each(|u| u.cast(dtype));
        self.resblocks.iter_mut().for_each(|b| b.cast(dtype));
        self.activation_post.cast(dtype);
        self.conv_post.cast(dtype);
    }

    /// Load a PyTorch-layout state dict (weight-normalised or fused), strictly:
    /// every tensor must be used and every parameter found.
    pub fn load_state_dict(&mut self, mut sd: StateDict) -> Result<()> {
        self.conv_pre.load_state(&mut sd, "conv_pre")?;
        // The reference wraps each upsampler in a one-element ModuleList.
        for (i, up) in self.ups.iter_mut().enumerate() {
            let nested = join_key(&format!("ups.{i}"), "0");
            let prefix = if sd.contains(&join_key(&nested, "weight_v"))
                || sd.contains(&join_key(&nested, "weight"))
            {
                nested
            } else {
                format!("ups.{i}")
            };
            up.load_state(&mut sd, &prefix)?;
        }
        for (i, block) in self.resblocks.iter_mut().enumerate() {
            block.load_state(&mut sd, &format!("resblocks.{i}"))?;
        }
        self.activation_post
            .load_state(&mut sd, "activation_post")?;
        self.conv_post.load_state(&mut sd, "conv_post")?;
        sd.finish()
    }

    /// The model's tensors under the reference's names and layout.
    pub fn state_dict(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        self.conv_pre.save_state("conv_pre", &mut out);
        for (i, up) in self.ups.iter().enumerate() {
            up.save_state(&format!("ups.{i}.0"), &mut out);
        }
        for (i, block) in self.resblocks.iter().enumerate() {
            block.save_state(&format!("resblocks.{i}"), &mut out);
        }
        self.activation_post.save_state("activation_post", &mut out);
        self.conv_post.save_state("conv_post", &mut out);
        out
    }

    /// Radius, in output samples, of the generator's receptive field: an
    /// output sample is unaffected by mel frames more than this many samples
    /// away (measured from the frame's start, `frame * hop_size`).
    pub fn receptive_radius_samples(&self) -> i32 {
        let num_kernels = self.num_kernels();
        // Work in samples at the current rate, one side. The input convolution
        // reaches `(7 - 1) / 2` frames.
        let mut radius = self.conv_pre.config().kernel_size / 2;
        for (i, up) in self.ups.iter().enumerate() {
            let (rate, kernel, pad) = up.geometry();
            // Output j reads inputs t with t*rate - pad <= j <= t*rate - pad + kernel - 1.
            radius = radius * rate + pad.max(kernel - 1 - pad);
            let blocks = &self.resblocks[i * num_kernels..(i + 1) * num_kernels];
            radius += blocks
                .iter()
                .map(AMPBlock::receptive_radius)
                .max()
                .unwrap_or(0);
        }
        radius += self.activation_post.receptive_radius();
        radius + self.conv_post.config().kernel_size / 2
    }

    /// The receptive-field radius in whole mel frames (rounded up).
    pub fn receptive_radius_frames(&self) -> i32 {
        let hop = self.config.hop_size;
        (self.receptive_radius_samples() + hop - 1) / hop
    }

    /// The activation in use.
    pub fn activation(&self) -> ActivationType {
        self.config.activation
    }
}

impl_module_params!(BigVGAN; conv_pre, ups, resblocks, activation_post, conv_post);
