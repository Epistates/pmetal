//! Neural network layers for BigVGAN.
//!
//! This module provides the core building blocks:
//! - Snake/SnakeBeta periodic activations
//! - Anti-aliased Activation1d wrapper
//! - AMP (Anti-aliased Multi-Periodicity) blocks
//! - Weight-normalized convolutions

mod activation;
mod amp_block;
mod conv;

pub use activation::{Activation1d, Snake, SnakeBeta};
pub use amp_block::{AMPBlock, AMPBlockSnake, ResidualBranch, ResidualBranchSnake};
pub use conv::{WeightNormConv1d, WeightNormConvTranspose1d};
