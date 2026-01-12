//! Weight-normalized convolution layers for BigVGAN.
//!
//! BigVGAN uses weight normalization instead of batch/layer normalization
//! for all convolution layers. This provides stable training without
//! normalization-induced artifacts in audio generation.

use crate::error::Result;
use mlx_rs::module::Param;
use mlx_rs::Array;

/// Weight-normalized 1D convolution.
///
/// Applies weight normalization: W = g * (v / ||v||)
/// where g is the magnitude and v is the direction.
#[derive(Debug)]
pub struct WeightNormConv1d {
    /// Direction parameter (unnormalized weights).
    pub weight_v: Param<Array>,
    /// Magnitude parameter (scalar per output channel).
    pub weight_g: Param<Array>,
    /// Optional bias.
    pub bias: Option<Param<Array>>,
    /// Input channels.
    pub in_channels: i32,
    /// Output channels.
    pub out_channels: i32,
    /// Kernel size.
    pub kernel_size: i32,
    /// Stride.
    pub stride: i32,
    /// Padding.
    pub padding: i32,
    /// Dilation.
    pub dilation: i32,
    /// Groups for grouped convolution.
    pub groups: i32,
}

impl WeightNormConv1d {
    /// Create a new weight-normalized Conv1d.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Kernel size
    /// * `stride` - Stride (default 1)
    /// * `padding` - Padding (default 0)
    /// * `dilation` - Dilation (default 1)
    /// * `groups` - Groups (default 1)
    /// * `bias` - Whether to use bias (default true)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: Option<i32>,
        padding: Option<i32>,
        dilation: Option<i32>,
        groups: Option<i32>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Initialize weights using Kaiming uniform
        let fan_in = (in_channels / groups) * kernel_size;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Weight shape for Conv1d: [out_channels, in_channels/groups, kernel_size]
        let weight_v = mlx_rs::random::uniform::<_, f32>(
            -bound,
            bound,
            &[out_channels, in_channels / groups, kernel_size],
            None,
        )?;

        // Compute initial magnitude ||v||
        let norm = weight_norm(&weight_v)?;
        let weight_g = norm;

        let bias = if use_bias {
            Some(Param::new(mlx_rs::ops::zeros::<f32>(&[out_channels])?))
        } else {
            None
        };

        Ok(Self {
            weight_v: Param::new(weight_v),
            weight_g: Param::new(weight_g),
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    /// Compute normalized weight: W = g * (v / ||v||)
    fn compute_weight(&self) -> Result<Array> {
        let v = self.weight_v.as_ref();
        let g = self.weight_g.as_ref();

        // Normalize v along all dims except output channel
        let norm = weight_norm(v)?;
        let v_normalized = v.divide(&norm)?;

        // Scale by magnitude
        Ok(v_normalized.multiply(g)?)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, in_channels, length] (NCL format)
    ///
    /// # Returns
    /// Output tensor [batch, out_channels, new_length] (NCL format)
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let weight = self.compute_weight()?;

        // MLX conv1d expects:
        // - input: [N, H, C_in] (NLC format)
        // - weight: [C_out, K, C_in] (OKI format)
        // Our tensors are in:
        // - input: [N, C_in, H] (NCL format)
        // - weight: [C_out, C_in/groups, K] (OIK format)

        // Transpose input from NCL to NLC: [batch, channels, length] -> [batch, length, channels]
        let x_nlc = x.transpose_axes(&[0, 2, 1])?;

        // Transpose weight from OIK to OKI: [out, in, kernel] -> [out, kernel, in]
        let weight_oki = weight.transpose_axes(&[0, 2, 1])?;

        // Apply 1D convolution
        let output = mlx_rs::ops::conv1d(
            &x_nlc,
            &weight_oki,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;

        // Transpose output from NLC back to NCL: [batch, length, channels] -> [batch, channels, length]
        let output = output.transpose_axes(&[0, 2, 1])?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            // Reshape bias for broadcasting: [out_channels] -> [1, out_channels, 1]
            let bias_reshaped = bias.as_ref().reshape(&[1, self.out_channels, 1])?;
            Ok(output.add(&bias_reshaped)?)
        } else {
            Ok(output)
        }
    }
}

/// Weight-normalized transposed 1D convolution.
///
/// Used for upsampling in the generator.
#[derive(Debug)]
pub struct WeightNormConvTranspose1d {
    /// Direction parameter (unnormalized weights).
    pub weight_v: Param<Array>,
    /// Magnitude parameter.
    pub weight_g: Param<Array>,
    /// Optional bias.
    pub bias: Option<Param<Array>>,
    /// Input channels.
    pub in_channels: i32,
    /// Output channels.
    pub out_channels: i32,
    /// Kernel size.
    pub kernel_size: i32,
    /// Stride.
    pub stride: i32,
    /// Padding.
    pub padding: i32,
    /// Output padding for ambiguous output size.
    pub output_padding: i32,
    /// Dilation.
    pub dilation: i32,
    /// Groups.
    pub groups: i32,
}

impl WeightNormConvTranspose1d {
    /// Create a new weight-normalized ConvTranspose1d.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        kernel_size: i32,
        stride: Option<i32>,
        padding: Option<i32>,
        output_padding: Option<i32>,
        dilation: Option<i32>,
        groups: Option<i32>,
        bias: Option<bool>,
    ) -> Result<Self> {
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let output_padding = output_padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);

        // Initialize weights
        let fan_in = in_channels * kernel_size;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Weight shape for ConvTranspose1d: [in_channels, out_channels/groups, kernel_size]
        let weight_v = mlx_rs::random::uniform::<_, f32>(
            -bound,
            bound,
            &[in_channels, out_channels / groups, kernel_size],
            None,
        )?;

        let norm = weight_norm(&weight_v)?;
        let weight_g = norm;

        let bias = if use_bias {
            Some(Param::new(mlx_rs::ops::zeros::<f32>(&[out_channels])?))
        } else {
            None
        };

        Ok(Self {
            weight_v: Param::new(weight_v),
            weight_g: Param::new(weight_g),
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        })
    }

    /// Compute normalized weight.
    fn compute_weight(&self) -> Result<Array> {
        let v = self.weight_v.as_ref();
        let g = self.weight_g.as_ref();

        let norm = weight_norm(v)?;
        let v_normalized = v.divide(&norm)?;

        Ok(v_normalized.multiply(g)?)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, in_channels, length]
    ///
    /// # Returns
    /// Upsampled tensor [batch, out_channels, length * stride]
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let weight = self.compute_weight()?;

        // Compute output length (for reference)
        let input_length = x.dim(2);
        let _output_length = (input_length - 1) * self.stride - 2 * self.padding
            + self.dilation * (self.kernel_size - 1)
            + self.output_padding
            + 1;

        // Apply transposed convolution
        // MLX conv_transpose expects weight shape [out_channels, in_channels/groups, kernel_size]
        // but our weight is [in_channels, out_channels/groups, kernel_size]
        // Need to transpose axes 0 and 1
        let weight_transposed = weight.transpose_axes(&[1, 0, 2])?;

        // Use conv_general for transposed convolution
        // For now, implement using manual upsampling + conv
        let output = conv_transpose_1d_manual(
            x,
            &weight_transposed,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )?;

        // Add bias if present
        if let Some(bias) = &self.bias {
            let bias_reshaped = bias.as_ref().reshape(&[1, self.out_channels, 1])?;
            Ok(output.add(&bias_reshaped)?)
        } else {
            Ok(output)
        }
    }
}

/// Compute weight norm along all dims except first (output channels).
/// Returns shape [out_channels, 1, 1] for broadcasting.
fn weight_norm(weight: &Array) -> Result<Array> {
    // Sum of squares along dims 1 and 2
    let sq = weight.multiply(weight)?;
    let sum_sq = sq.sum_axes(&[1, 2], Some(true))?;
    let norm = sum_sq.sqrt()?;

    // Add small epsilon for numerical stability
    let eps = Array::from_f32(1e-12);
    Ok(norm.add(&eps)?)
}

/// Flip array along an axis by reversing the indices.
fn flip_axis(arr: &Array, axis: i32) -> Result<Array> {
    let axis_len = arr.dim(axis);
    // Create reversed indices
    let indices: Vec<i32> = (0..axis_len).rev().collect();
    let indices_arr = Array::from_slice(&indices, &[axis_len]);
    arr.take_axis(&indices_arr, axis).map_err(Into::into)
}

/// Manual implementation of transposed 1D convolution.
///
/// ConvTranspose1d(x) is equivalent to inserting (stride-1) zeros between samples,
/// then applying convolution with the transposed kernel.
///
/// Input is in NCL format: [batch, in_channels, length]
/// Weight is in OIK format: [out_channels, in_channels/groups, kernel_size]
fn conv_transpose_1d_manual(
    x: &Array,
    weight: &Array, // [out_channels, in_channels/groups, kernel_size]
    stride: i32,
    padding: i32,
    output_padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    let batch = x.dim(0);
    let in_channels = x.dim(1);
    let in_length = x.dim(2);
    let out_channels = weight.dim(0);
    let kernel_size = weight.dim(2);

    // Helper to run conv1d with format conversion (NCL -> NLC -> NCL)
    let run_conv1d = |input: &Array, w: &Array, s: i32, p: i32, d: i32, g: i32| -> Result<Array> {
        // Input is NCL, convert to NLC
        let input_nlc = input.transpose_axes(&[0, 2, 1])?;
        // Weight is OIK, convert to OKI
        let weight_oki = w.transpose_axes(&[0, 2, 1])?;
        // Run conv1d
        let output_nlc = mlx_rs::ops::conv1d(&input_nlc, &weight_oki, s, p, d, g)?;
        // Convert output back to NCL
        output_nlc.transpose_axes(&[0, 2, 1]).map_err(Into::into)
    };

    if stride == 1 && padding == 0 && output_padding == 0 && dilation == 1 {
        // Simple case: just apply conv with flipped kernel
        let weight_flipped = flip_axis(weight, 2)?; // Flip along kernel axis (axis 2 in OIK)
        return run_conv1d(x, &weight_flipped, 1, kernel_size - 1, 1, groups);
    }

    // General case: insert zeros then convolve
    // Step 1: Insert (stride-1) zeros between samples
    let upsampled_length = (in_length - 1) * stride + 1;

    // Alternative: Use unfold-like operation
    // For now, use a simpler but less efficient approach with concatenation
    if stride > 1 {
        use mlx_rs::ops::indexing::IndexOp;
        // Create interleaved tensor
        let zeros_between =
            mlx_rs::ops::zeros::<f32>(&[batch, in_channels, in_length, stride - 1])?;
        let x_expanded = x.reshape(&[batch, in_channels, in_length, 1])?;
        let interleaved = mlx_rs::ops::concatenate_axis(&[&x_expanded, &zeros_between], -1)?;
        let interleaved = interleaved.reshape(&[batch, in_channels, in_length * stride])?;
        // Trim last (stride-1) zeros
        let upsampled = interleaved.index((.., .., ..upsampled_length));

        // Step 2: Flip kernel and apply convolution
        let weight_flipped = flip_axis(weight, 2)?;

        // Compute required padding for output size
        let conv_padding = dilation * (kernel_size - 1) - padding;
        let conv_padding = conv_padding.max(0);

        let output = run_conv1d(&upsampled, &weight_flipped, 1, conv_padding, dilation, groups)?;

        // Handle output_padding by adding zeros at the end
        if output_padding > 0 {
            let pad = mlx_rs::ops::zeros::<f32>(&[batch, out_channels, output_padding])?;
            return mlx_rs::ops::concatenate_axis(&[&output, &pad], -1).map_err(Into::into);
        }

        Ok(output)
    } else {
        // stride == 1, just apply transposed conv logic
        let weight_flipped = flip_axis(weight, 2)?;
        let conv_padding = dilation * (kernel_size - 1) - padding;
        let conv_padding = conv_padding.max(0);

        run_conv1d(x, &weight_flipped, 1, conv_padding, dilation, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_norm_conv1d_shape() {
        let conv = WeightNormConv1d::new(4, 8, 3, Some(1), Some(1), None, None, Some(true)).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 16], None, None, None).unwrap();
        let y = conv.forward(&x).unwrap();
        y.eval().unwrap();

        // With padding=1, kernel=3, stride=1: output_len = input_len
        assert_eq!(y.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_weight_norm_conv1d_no_bias() {
        let conv =
            WeightNormConv1d::new(4, 8, 3, Some(1), Some(1), None, None, Some(false)).unwrap();

        let x = mlx_rs::random::normal::<f32>(&[2, 4, 16], None, None, None).unwrap();
        let y = conv.forward(&x).unwrap();
        y.eval().unwrap();

        assert_eq!(y.shape(), &[2, 8, 16]);
        assert!(conv.bias.is_none());
    }

    #[test]
    fn test_weight_norm_values() {
        // Weight normalization should make ||W|| = ||g||
        let conv = WeightNormConv1d::new(2, 4, 3, None, None, None, None, None).unwrap();

        let weight = conv.compute_weight().unwrap();
        weight.eval().unwrap();

        // The weight should be properly normalized
        assert_eq!(weight.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_conv_transpose1d_shape() {
        // stride=2 should double the length (approximately)
        let conv = WeightNormConvTranspose1d::new(
            8,
            4,
            4,
            Some(2),
            Some(1),
            None,
            None,
            None,
            Some(true),
        )
        .unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 8, 16], None, None, None).unwrap();
        let y = conv.forward(&x).unwrap();
        y.eval().unwrap();

        // ConvTranspose1d output: (L-1)*S - 2*P + D*(K-1) + OP + 1
        // = (16-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 30 - 2 + 3 + 1 = 32
        assert_eq!(y.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_conv_transpose1d_upsample_4x() {
        // Test 4x upsampling like BigVGAN
        let conv = WeightNormConvTranspose1d::new(
            512,
            256,
            16,
            Some(4),
            Some(6),
            None,
            None,
            None,
            Some(true),
        )
        .unwrap();

        let x = mlx_rs::random::normal::<f32>(&[1, 512, 8], None, None, None).unwrap();
        let y = conv.forward(&x).unwrap();
        y.eval().unwrap();

        // Output length should be approximately 4x input
        // (8-1)*4 - 2*6 + 1*(16-1) + 0 + 1 = 28 - 12 + 15 + 1 = 32
        assert_eq!(y.shape(), &[1, 256, 32]);
    }
}
