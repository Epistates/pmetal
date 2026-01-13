//! Anti-aliased Multi-Periodicity (AMP) blocks for BigVGAN.
//!
//! AMP blocks combine multiple residual branches with different dilation rates
//! and kernel sizes to capture audio signals at multiple periodicities.
//! Anti-aliased activations prevent aliasing artifacts.

use crate::error::Result;
use crate::nn::{Activation1d, Snake, SnakeBeta, WeightNormConv1d};
use mlx_rs::Array;

/// Anti-aliased Multi-Periodicity (AMP) block.
///
/// Each AMP block consists of multiple residual branches, each with
/// different dilation patterns to capture different periodicities in
/// the audio signal.
#[derive(Debug)]
pub struct AMPBlock {
    /// Number of channels.
    pub channels: i32,
    /// Kernel sizes for each branch.
    pub kernel_sizes: Vec<i32>,
    /// Dilation rates for each branch and layer.
    pub dilations: Vec<Vec<i32>>,
    /// Residual branches (each branch has multiple conv layers).
    pub branches: Vec<ResidualBranch>,
}

/// A single residual branch within an AMP block.
#[derive(Debug)]
pub struct ResidualBranch {
    /// Convolution layers with activations.
    pub layers: Vec<(Activation1d<SnakeBeta>, WeightNormConv1d)>,
}

impl AMPBlock {
    /// Create a new AMP block.
    ///
    /// # Arguments
    /// * `channels` - Number of input/output channels
    /// * `kernel_size` - Kernel size for convolutions
    /// * `dilations` - Dilation rates [[d1, d2, ...], [d1, d2, ...], ...]
    ///   Each inner vec is a branch, each element is a dilation rate for a layer
    pub fn new(channels: i32, kernel_size: i32, dilations: Vec<Vec<i32>>) -> Result<Self> {
        let mut branches = Vec::with_capacity(dilations.len());

        for branch_dilations in &dilations {
            let mut layers = Vec::with_capacity(branch_dilations.len());

            for &dilation in branch_dilations {
                // Anti-aliased Snake activation
                let activation = SnakeBeta::new(channels, true)?;
                let act1d = Activation1d::new(activation, Some(2), Some(2))?;

                // Weight-normalized convolution with dilation
                let padding = (kernel_size - 1) * dilation / 2;
                let conv = WeightNormConv1d::new(
                    channels,
                    channels,
                    kernel_size,
                    Some(1),
                    Some(padding),
                    Some(dilation),
                    None,
                    Some(true),
                )?;

                layers.push((act1d, conv));
            }

            branches.push(ResidualBranch { layers });
        }

        let kernel_sizes = vec![kernel_size; dilations.len()];

        Ok(Self {
            channels,
            kernel_sizes,
            dilations,
            branches,
        })
    }

    /// Create an AMP block with BigVGAN v2 configuration.
    ///
    /// Default: kernel_size=3, dilations=[[1,3,5], [1,3,5], [1,3,5]]
    pub fn bigvgan_v2(channels: i32) -> Result<Self> {
        Self::new(
            channels,
            3,
            vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        )
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, channels, time]
    ///
    /// # Returns
    /// Output tensor with same shape
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Sum outputs from all branches
        let mut output: Option<Array> = None;

        for branch in &self.branches {
            let mut branch_out = x.clone();

            for (activation, conv) in &branch.layers {
                // Apply anti-aliased activation then convolution
                branch_out = activation.forward(&branch_out)?;
                branch_out = conv.forward(&branch_out)?;
            }

            // Add to cumulative output
            match &output {
                Some(o) => output = Some(o.add(&branch_out)?),
                None => output = Some(branch_out),
            }
        }

        // Average over branches and add residual
        let num_branches = Array::from_int(self.branches.len() as i32);
        let branch_avg = output.unwrap().divide(&num_branches)?;

        Ok(x.add(&branch_avg)?)
    }
}

/// Simplified AMP block using Snake (single alpha parameter).
#[derive(Debug)]
pub struct AMPBlockSnake {
    /// Number of channels.
    pub channels: i32,
    /// Residual branches.
    pub branches: Vec<ResidualBranchSnake>,
}

/// Residual branch using Snake activation.
#[derive(Debug)]
pub struct ResidualBranchSnake {
    /// Layers with Snake activation.
    pub layers: Vec<(Activation1d<Snake>, WeightNormConv1d)>,
}

impl AMPBlockSnake {
    /// Create a new AMP block with Snake activation.
    pub fn new(channels: i32, kernel_size: i32, dilations: Vec<Vec<i32>>) -> Result<Self> {
        let mut branches = Vec::with_capacity(dilations.len());

        for branch_dilations in &dilations {
            let mut layers = Vec::with_capacity(branch_dilations.len());

            for &dilation in branch_dilations {
                let activation = Snake::new(channels, true)?;
                let act1d = Activation1d::new(activation, Some(2), Some(2))?;

                let padding = (kernel_size - 1) * dilation / 2;
                let conv = WeightNormConv1d::new(
                    channels,
                    channels,
                    kernel_size,
                    Some(1),
                    Some(padding),
                    Some(dilation),
                    None,
                    Some(true),
                )?;

                layers.push((act1d, conv));
            }

            branches.push(ResidualBranchSnake { layers });
        }

        Ok(Self { channels, branches })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut output: Option<Array> = None;

        for branch in &self.branches {
            let mut branch_out = x.clone();

            for (activation, conv) in &branch.layers {
                branch_out = activation.forward(&branch_out)?;
                branch_out = conv.forward(&branch_out)?;
            }

            match &output {
                Some(o) => output = Some(o.add(&branch_out)?),
                None => output = Some(branch_out),
            }
        }

        let num_branches = Array::from_int(self.branches.len() as i32);
        let branch_avg = output.unwrap().divide(&num_branches)?;

        Ok(x.add(&branch_avg)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amp_block_shape() {
        let amp = AMPBlock::new(64, 3, vec![vec![1, 3, 5]]).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 64, 128], None, None, None).unwrap();
        let y = amp.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[1, 64, 128]);
    }

    #[test]
    fn test_amp_block_bigvgan_v2() {
        let amp = AMPBlock::bigvgan_v2(128).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 128, 64], None, None, None).unwrap();
        let y = amp.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[2, 128, 64]);
        assert_eq!(amp.branches.len(), 3);
    }

    #[test]
    fn test_amp_block_residual() {
        // Verify residual connection works
        let amp = AMPBlock::new(32, 3, vec![vec![1]]).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 32, 16], None, None, None).unwrap();
        let y = amp.forward(&x).unwrap();
        x.eval().unwrap();
        y.eval().unwrap();

        // Output should be different from input (processed) but same shape
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_amp_block_snake() {
        let amp = AMPBlockSnake::new(64, 3, vec![vec![1, 3], vec![1, 3]]).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 64, 32], None, None, None).unwrap();
        let y = amp.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[1, 64, 32]);
    }

    #[test]
    fn test_amp_block_multiple_branches() {
        // Test with 4 branches like some BigVGAN configs
        let amp =
            AMPBlock::new(256, 3, vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8]]).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 256, 64], None, None, None).unwrap();
        let y = amp.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[1, 256, 64]);
        assert_eq!(amp.branches.len(), 4);
    }
}
