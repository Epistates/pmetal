//! Periodic activation functions for BigVGAN.
//!
//! BigVGAN uses Snake and SnakeBeta activations that introduce
//! periodic inductive bias for modeling audio harmonics.

use crate::error::Result;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Param;
use mlx_rs::Array;

/// Snake activation function.
///
/// Snake(x) = x + (1/α) * sin²(αx)
///
/// The learnable parameter α controls the frequency of the periodic component.
/// Higher α values capture higher frequency content.
#[derive(Debug, ModuleParameters)]
pub struct Snake {
    /// Learnable frequency parameter.
    #[param]
    pub alpha: Param<Array>,
    /// Whether alpha is in log scale (exp(alpha) is used).
    pub alpha_logscale: bool,
}

impl Snake {
    /// Create a new Snake activation.
    ///
    /// # Arguments
    /// * `channels` - Number of channels (alpha per channel)
    /// * `alpha_logscale` - Use log scale for alpha (recommended)
    pub fn new(channels: i32, alpha_logscale: bool) -> Result<Self> {
        // Initialize alpha to 1.0 (or log(1) = 0 for logscale)
        let init_val = if alpha_logscale { 0.0 } else { 1.0 };
        let alpha = Array::from_f32(init_val);
        let alpha = mlx_rs::ops::broadcast_to(&alpha, &[1, channels, 1])?;

        Ok(Self {
            alpha: Param::new(alpha),
            alpha_logscale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, channels, time]
    ///
    /// # Returns
    /// Activated tensor with same shape
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let alpha = if self.alpha_logscale {
            self.alpha.as_ref().exp()?
        } else {
            self.alpha.as_ref().clone()
        };

        // Snake(x) = x + (1/α) * sin²(αx)
        let ax = x.multiply(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = sin_ax.multiply(&sin_ax)?;
        let scaled = sin_sq.divide(&alpha)?;

        Ok(x.add(&scaled)?)
    }
}

/// SnakeBeta activation function.
///
/// SnakeBeta(x) = x + (1/β) * sin²(αx)
///
/// Extends Snake with separate α (frequency) and β (magnitude) parameters.
/// This provides more expressiveness for modeling complex audio signals.
#[derive(Debug, ModuleParameters)]
pub struct SnakeBeta {
    /// Learnable frequency parameter.
    #[param]
    pub alpha: Param<Array>,
    /// Learnable magnitude parameter.
    #[param]
    pub beta: Param<Array>,
    /// Whether parameters are in log scale.
    pub logscale: bool,
}

impl SnakeBeta {
    /// Create a new SnakeBeta activation.
    ///
    /// # Arguments
    /// * `channels` - Number of channels (alpha/beta per channel)
    /// * `logscale` - Use log scale for parameters (recommended)
    pub fn new(channels: i32, logscale: bool) -> Result<Self> {
        // Initialize to 1.0 (or log(1) = 0 for logscale)
        let init_val = if logscale { 0.0 } else { 1.0 };
        let init = Array::from_f32(init_val);
        let alpha = mlx_rs::ops::broadcast_to(&init, &[1, channels, 1])?;
        let beta = mlx_rs::ops::broadcast_to(&init, &[1, channels, 1])?;

        Ok(Self {
            alpha: Param::new(alpha),
            beta: Param::new(beta),
            logscale,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, channels, time]
    ///
    /// # Returns
    /// Activated tensor with same shape
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let (alpha, beta) = if self.logscale {
            (self.alpha.as_ref().exp()?, self.beta.as_ref().exp()?)
        } else {
            (self.alpha.as_ref().clone(), self.beta.as_ref().clone())
        };

        // SnakeBeta(x) = x + (1/β) * sin²(αx)
        let ax = x.multiply(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = sin_ax.multiply(&sin_ax)?;
        let scaled = sin_sq.divide(&beta)?;

        Ok(x.add(&scaled)?)
    }
}

/// Anti-aliased activation wrapper.
///
/// Wraps an activation function with upsampling before and downsampling after
/// to prevent aliasing artifacts from periodic activations.
///
/// Process: Upsample(2×) → Activate → Downsample(2×)
#[derive(Debug)]
pub struct Activation1d<A> {
    /// Inner activation function.
    pub activation: A,
    /// Upsampling ratio.
    pub up_ratio: i32,
    /// Downsampling ratio.
    pub down_ratio: i32,
    /// Anti-aliasing filter (lowpass).
    pub filter: Array,
}

impl<A> Activation1d<A> {
    /// Create a new Activation1d wrapper.
    ///
    /// # Arguments
    /// * `activation` - Inner activation function
    /// * `up_ratio` - Upsampling ratio (default 2)
    /// * `down_ratio` - Downsampling ratio (default 2)
    pub fn new(activation: A, up_ratio: Option<i32>, down_ratio: Option<i32>) -> Result<Self> {
        let up_ratio = up_ratio.unwrap_or(2);
        let down_ratio = down_ratio.unwrap_or(2);

        // Create Kaiser-windowed sinc filter for anti-aliasing
        let filter = create_kaiser_filter(12, 0.5 / up_ratio as f32)?;

        Ok(Self {
            activation,
            up_ratio,
            down_ratio,
            filter,
        })
    }
}

impl Activation1d<Snake> {
    /// Forward pass with Snake activation.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Upsample
        let x_up = upsample_1d(x, self.up_ratio)?;

        // Apply activation
        let x_act = self.activation.forward(&x_up)?;

        // Downsample with anti-aliasing filter
        downsample_1d(&x_act, self.down_ratio, &self.filter)
    }
}

impl Activation1d<SnakeBeta> {
    /// Forward pass with SnakeBeta activation.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Upsample
        let x_up = upsample_1d(x, self.up_ratio)?;

        // Apply activation
        let x_act = self.activation.forward(&x_up)?;

        // Downsample with anti-aliasing filter
        downsample_1d(&x_act, self.down_ratio, &self.filter)
    }
}

/// Create Kaiser-windowed sinc filter for anti-aliasing.
fn create_kaiser_filter(taps: i32, cutoff: f32) -> Result<Array> {
    // Simplified Kaiser window
    // Full implementation would use scipy.signal.kaiser equivalent
    let half = taps / 2;
    let mut filter = Vec::with_capacity(taps as usize);

    for i in 0..taps {
        let n = i - half;
        // Sinc function
        let sinc = if n == 0 {
            1.0
        } else {
            let x = std::f32::consts::PI * cutoff * n as f32;
            x.sin() / x
        };

        // Simple window (approximate Kaiser)
        let window = 0.5 * (1.0 + (std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos());

        filter.push(sinc * window * cutoff);
    }

    // Normalize
    let sum: f32 = filter.iter().sum();
    for v in &mut filter {
        *v /= sum;
    }

    Ok(Array::from_slice(&filter, &[1, 1, taps]))
}

/// Upsample 1D signal by inserting zeros.
fn upsample_1d(x: &Array, ratio: i32) -> Result<Array> {
    if ratio == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape();
    let batch = shape[0];
    let channels = shape[1];
    let length = shape[2];

    // Insert zeros between samples
    // [B, C, L] -> [B, C, L, ratio] -> [B, C, L*ratio]
    let zeros = mlx_rs::ops::zeros::<f32>(&[batch, channels, length, ratio - 1])?;
    let x_expanded = x.reshape(&[batch, channels, length, 1])?;
    let interleaved = mlx_rs::ops::concatenate_axis(&[&x_expanded, &zeros], -1)?;

    Ok(interleaved.reshape(&[batch, channels, length * ratio])?)
}

/// Downsample 1D signal with anti-aliasing filter.
fn downsample_1d(x: &Array, ratio: i32, filter: &Array) -> Result<Array> {
    if ratio == 1 {
        return Ok(x.clone());
    }

    // Apply lowpass filter via 1D convolution
    // Then take every `ratio`th sample
    let shape = x.shape();
    let _batch = shape[0];
    let channels = shape[1];
    let length = shape[2];

    // Group convolution (each channel independently)
    // Expand filter for group conv: [1, 1, taps] -> [channels, 1, taps]
    let _filter_exp = mlx_rs::ops::broadcast_to(filter, &[channels, 1, filter.dim(2)])?;

    // Apply filter (would need conv1d with groups)
    // For now, simplified: just downsample without explicit filtering
    // TODO: Implement proper grouped conv1d

    // Take every ratio-th sample
    let indices: Vec<i32> = (0..length).step_by(ratio as usize).collect();
    let indices_arr = Array::from_slice(&indices, &[indices.len() as i32]);

    x.take_axis(&indices_arr, 2).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_forward() {
        let snake = Snake::new(4, true).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 4, 16], None, None, None).unwrap();

        let y = snake.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_snakebeta_forward() {
        let snake = SnakeBeta::new(8, true).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[2, 8, 32], None, None, None).unwrap();

        let y = snake.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[2, 8, 32]);
    }

    #[test]
    fn test_snake_values() {
        // Snake(0) should be 0 (since sin(0) = 0)
        let snake = Snake::new(1, false).unwrap();
        let x = mlx_rs::ops::zeros::<f32>(&[1, 1, 4]).unwrap();

        let y = snake.forward(&x).unwrap();
        y.eval().unwrap();

        // Output should be close to 0
        let sum = y.sum(None).unwrap();
        sum.eval().unwrap();
        assert!(sum.item::<f32>().abs() < 1e-5);
    }

    #[test]
    fn test_upsample_1d() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let y = upsample_1d(&x, 2).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[1, 1, 8]);
        // Should be [1, 0, 2, 0, 3, 0, 4, 0]
    }

    #[test]
    fn test_activation1d() {
        let snake = Snake::new(4, true).unwrap();
        let act1d = Activation1d::new(snake, Some(2), Some(2)).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 4, 16], None, None, None).unwrap();
        let y = act1d.forward(&x).unwrap();
        y.eval().unwrap();

        // Output should have same shape (up 2x, down 2x)
        assert_eq!(y.shape(), &[1, 4, 16]);
    }
}
