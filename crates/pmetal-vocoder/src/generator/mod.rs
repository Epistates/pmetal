//! BigVGAN Generator for mel-to-waveform synthesis.
//!
//! The generator converts mel spectrograms to high-fidelity audio waveforms
//! using transposed convolutions for upsampling and AMP blocks for processing.

use crate::config::BigVGANConfig;
use crate::error::{Result, VocoderError};
use crate::nn::{Activation1d, AMPBlock, SnakeBeta, WeightNormConv1d, WeightNormConvTranspose1d};
use mlx_rs::Array;
use std::path::Path;

/// BigVGAN neural vocoder.
///
/// Converts mel spectrograms [batch, n_mels, frames] to audio waveforms
/// [batch, 1, samples]. Uses anti-aliased upsampling with AMP blocks.
#[derive(Debug)]
pub struct BigVGAN {
    /// Model configuration.
    pub config: BigVGANConfig,
    /// Initial convolution from mel channels to hidden.
    pub conv_pre: WeightNormConv1d,
    /// Upsampling stages (transposed convolutions).
    pub upsamples: Vec<WeightNormConvTranspose1d>,
    /// AMP blocks after each upsample stage.
    pub amp_blocks: Vec<Vec<AMPBlock>>,
    /// Final activation.
    pub activation_post: Activation1d<SnakeBeta>,
    /// Final convolution to single channel audio.
    pub conv_post: WeightNormConv1d,
}

impl BigVGAN {
    /// Create a new BigVGAN generator from configuration.
    pub fn new(config: BigVGANConfig) -> Result<Self> {
        let num_upsamples = config.upsample_rates.len();
        let num_kernels = config.resblock_kernel_sizes.len();

        // Initial convolution: n_mels -> initial_channel
        let conv_pre = WeightNormConv1d::new(
            config.num_mels,
            config.upsample_initial_channel,
            7,
            Some(1),
            Some(3),
            None,
            None,
            Some(true),
        )?;

        // Build upsampling stages
        let mut upsamples = Vec::with_capacity(num_upsamples);
        let mut amp_blocks = Vec::with_capacity(num_upsamples);

        let mut channels = config.upsample_initial_channel;

        for i in 0..num_upsamples {
            let out_channels = channels / 2;
            let upsample_rate = config.upsample_rates[i];
            let kernel_size = config.upsample_kernel_sizes[i];
            let padding = (kernel_size - upsample_rate) / 2;

            // Transposed convolution for upsampling
            let upsample = WeightNormConvTranspose1d::new(
                channels,
                out_channels,
                kernel_size,
                Some(upsample_rate),
                Some(padding),
                None,
                None,
                None,
                Some(true),
            )?;
            upsamples.push(upsample);

            // AMP blocks after upsampling
            let mut stage_amps = Vec::with_capacity(num_kernels);
            for j in 0..num_kernels {
                let kernel_size = config.resblock_kernel_sizes[j];
                // Each dilation list becomes a single branch in AMP block
                // Wrap it in a vec to create one branch
                let dilations = vec![config.resblock_dilation_sizes[j].clone()];
                let amp = AMPBlock::new(out_channels, kernel_size, dilations)?;
                stage_amps.push(amp);
            }
            amp_blocks.push(stage_amps);

            channels = out_channels;
        }

        // Final activation and output convolution
        let activation_post = Activation1d::new(SnakeBeta::new(channels, true)?, Some(2), Some(2))?;

        let conv_post = WeightNormConv1d::new(
            channels,
            1, // mono audio output
            7,
            Some(1),
            Some(3),
            None,
            None,
            Some(true),
        )?;

        Ok(Self {
            config,
            conv_pre,
            upsamples,
            amp_blocks,
            activation_post,
            conv_post,
        })
    }

    /// Create BigVGAN v2 24kHz 100-band model.
    pub fn v2_24khz_100band() -> Result<Self> {
        Self::new(BigVGANConfig::v2_24khz_100band())
    }

    /// Create BigVGAN v2 44.1kHz 128-band model.
    pub fn v2_44khz_128band() -> Result<Self> {
        Self::new(BigVGANConfig::v2_44khz_128band())
    }

    /// Load pretrained weights from safetensors file.
    pub fn load_weights(&mut self, path: &Path) -> Result<()> {
        let file_data = std::fs::read(path).map_err(VocoderError::from)?;
        let tensors = safetensors::SafeTensors::deserialize(&file_data)
            .map_err(|e| VocoderError::WeightLoad(e.to_string()))?;

        // Load conv_pre weights
        if let Ok(weight_v) = tensors.tensor("conv_pre.weight_v") {
            let arr = tensor_to_array(weight_v)?;
            self.conv_pre.weight_v.value = arr;
        }
        if let Ok(weight_g) = tensors.tensor("conv_pre.weight_g") {
            let arr = tensor_to_array(weight_g)?;
            self.conv_pre.weight_g.value = arr;
        }
        if let Ok(bias) = tensors.tensor("conv_pre.bias") {
            if let Some(ref mut b) = self.conv_pre.bias {
                let arr = tensor_to_array(bias)?;
                b.value = arr;
            }
        }

        // Load upsample weights (TODO: implement fully)
        for (i, upsample) in self.upsamples.iter_mut().enumerate() {
            let prefix = format!("ups.{}", i);
            if let Ok(weight_v) = tensors.tensor(&format!("{}.weight_v", prefix)) {
                let arr = tensor_to_array(weight_v)?;
                upsample.weight_v.value = arr;
            }
            if let Ok(weight_g) = tensors.tensor(&format!("{}.weight_g", prefix)) {
                let arr = tensor_to_array(weight_g)?;
                upsample.weight_g.value = arr;
            }
        }

        // Load conv_post weights
        if let Ok(weight_v) = tensors.tensor("conv_post.weight_v") {
            let arr = tensor_to_array(weight_v)?;
            self.conv_post.weight_v.value = arr;
        }
        if let Ok(weight_g) = tensors.tensor("conv_post.weight_g") {
            let arr = tensor_to_array(weight_g)?;
            self.conv_post.weight_g.value = arr;
        }
        if let Ok(bias) = tensors.tensor("conv_post.bias") {
            if let Some(ref mut b) = self.conv_post.bias {
                let arr = tensor_to_array(bias)?;
                b.value = arr;
            }
        }

        Ok(())
    }

    /// Load pretrained model from HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        use hf_hub::api::sync::Api;

        let api = Api::new().map_err(|e| VocoderError::Hub(e.to_string()))?;
        let repo = api.model(model_id.to_string());

        // Download config
        let config_path = repo
            .get("config.json")
            .map_err(|e| VocoderError::Hub(e.to_string()))?;
        let config_str =
            std::fs::read_to_string(&config_path).map_err(VocoderError::from)?;
        let config: BigVGANConfig =
            serde_json::from_str(&config_str).map_err(|e| VocoderError::Config(e.to_string()))?;

        // Create model
        let mut model = Self::new(config)?;

        // Download and load weights
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| VocoderError::Hub(e.to_string()))?;
        model.load_weights(&weights_path)?;

        Ok(model)
    }

    /// Forward pass: convert mel spectrogram to audio.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, frames]
    ///
    /// # Returns
    /// Audio waveform [batch, 1, samples] normalized to [-1, 1]
    pub fn forward(&self, mel: &Array) -> Result<Array> {
        // Initial convolution
        let mut x = self.conv_pre.forward(mel)?;

        // Upsampling stages with AMP blocks
        for (i, upsample) in self.upsamples.iter().enumerate() {
            x = upsample.forward(&x)?;

            // Apply AMP blocks and sum their outputs
            let mut amp_out: Option<Array> = None;
            for amp in &self.amp_blocks[i] {
                let out = amp.forward(&x)?;
                match &amp_out {
                    Some(o) => amp_out = Some(o.add(&out)?),
                    None => amp_out = Some(out),
                }
            }

            // Average AMP outputs
            let num_amps = Array::from_int(self.amp_blocks[i].len() as i32);
            x = amp_out.unwrap().divide(&num_amps)?;
        }

        // Final activation and output convolution
        x = self.activation_post.forward(&x)?;
        x = self.conv_post.forward(&x)?;

        // Tanh to normalize to [-1, 1]
        Ok(mlx_rs::ops::tanh(&x)?)
    }

    /// Generate audio from mel spectrogram (inference helper).
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [n_mels, frames] or [batch, n_mels, frames]
    ///
    /// # Returns
    /// Audio samples [samples] or [batch, samples]
    pub fn generate(&self, mel: &Array) -> Result<Array> {
        let (mel, was_2d) = if mel.ndim() == 2 {
            (mel.reshape(&[1, mel.dim(0), mel.dim(1)])?, true)
        } else {
            (mel.clone(), false)
        };

        let audio = self.forward(&mel)?;

        // Remove channel dimension and optionally batch dimension
        let audio = audio.squeeze()?; // Remove channel dim

        if was_2d {
            Ok(audio.squeeze()?) // Remove batch dim
        } else {
            Ok(audio)
        }
    }
}

/// Convert safetensors tensor to MLX Array.
fn tensor_to_array(tensor: safetensors::tensor::TensorView<'_>) -> Result<Array> {
    let shape: Vec<i32> = tensor.shape().iter().map(|&s| s as i32).collect();
    let data = tensor.data();

    // Handle different dtypes
    match tensor.dtype() {
        safetensors::Dtype::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(data);
            Ok(Array::from_slice(floats, &shape))
        }
        safetensors::Dtype::F16 => {
            // Convert f16 to f32
            let f16s: &[half::f16] = bytemuck::cast_slice(data);
            let floats: Vec<f32> = f16s.iter().map(|f| f.to_f32()).collect();
            Ok(Array::from_slice(&floats, &shape))
        }
        safetensors::Dtype::BF16 => {
            // Convert bf16 to f32
            let bf16s: &[half::bf16] = bytemuck::cast_slice(data);
            let floats: Vec<f32> = bf16s.iter().map(|f| f.to_f32()).collect();
            Ok(Array::from_slice(&floats, &shape))
        }
        _ => Err(VocoderError::WeightLoad(format!(
            "Unsupported dtype: {:?}",
            tensor.dtype()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigvgan_new() {
        let config = BigVGANConfig::v2_24khz_100band();
        let model = BigVGAN::new(config).unwrap();

        assert_eq!(model.upsamples.len(), 6);
        assert_eq!(model.amp_blocks.len(), 6);
    }

    #[test]
    fn test_bigvgan_forward_shape() {
        let config = BigVGANConfig::v2_24khz_100band();
        let model = BigVGAN::new(config).unwrap();

        // Create a small mel spectrogram
        let mel = mlx_rs::random::normal::<f32>(&[1, 100, 10], None, None, None).unwrap();
        let audio = model.forward(&mel).unwrap();
        audio.eval().unwrap();

        // Output should be [1, 1, samples]
        // With 256x upsampling: 10 frames * 256 = 2560 samples
        assert_eq!(audio.dim(0), 1); // batch
        assert_eq!(audio.dim(1), 1); // mono channel
        // Note: exact length may vary due to conv padding
    }

    #[test]
    fn test_bigvgan_generate() {
        let config = BigVGANConfig::base_24khz_100band();
        let model = BigVGAN::new(config).unwrap();

        // Test 2D input (no batch)
        let mel = mlx_rs::random::normal::<f32>(&[100, 8], None, None, None).unwrap();
        let audio = model.generate(&mel).unwrap();
        audio.eval().unwrap();

        // Should be 1D output
        assert_eq!(audio.ndim(), 1);
    }

    #[test]
    fn test_bigvgan_output_range() {
        let config = BigVGANConfig::base_24khz_100band();
        let model = BigVGAN::new(config).unwrap();

        let mel = mlx_rs::random::normal::<f32>(&[1, 100, 4], None, None, None).unwrap();
        let audio = model.forward(&mel).unwrap();
        audio.eval().unwrap();

        // Output should be in [-1, 1] due to tanh
        let max_val = audio.max(None).unwrap();
        let min_val = audio.min(None).unwrap();
        max_val.eval().unwrap();
        min_val.eval().unwrap();

        assert!(max_val.item::<f32>() <= 1.0);
        assert!(min_val.item::<f32>() >= -1.0);
    }
}
