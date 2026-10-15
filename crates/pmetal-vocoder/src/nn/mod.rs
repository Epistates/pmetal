//! Neural network layers for BigVGAN.
//!
//! The building blocks, all channels-last (`[batch, time, channels]`):
//! - Snake / SnakeBeta periodic activations
//! - Kaiser-sinc up- and down-samplers and the anti-aliased [`Activation1d`]
//! - AMP (Anti-aliased Multi-Periodicity) residual blocks
//! - Weight-normalised convolutions ([`Conv1d`], [`ConvTranspose1d`], [`Conv2d`])

mod activation;
mod amp_block;
mod conv;
pub mod filter;
mod overlap;
mod resample;
mod weight_norm;

pub use activation::{Activation1d, PeriodicActivation, Snake, SnakeBeta};
pub use amp_block::AMPBlock;
pub use conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, ConvTranspose1d};
pub use overlap::overlap_add;
pub use resample::{DownSample1d, UpSample1d};
pub use weight_norm::{NormedWeight, StateDict, join_key};
