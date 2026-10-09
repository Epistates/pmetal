//! VAE (Variational Autoencoder) architectures.
//!
//! Implementation of Flux.1 and SDXL compatible VAEs optimized for Apple Silicon.
//! This implementation uses NHWC format for consistency with MLX standards.
use pmetal_bridge::compat::{Array, Exception, nn};
use pmetal_bridge::impl_module_params;

use serde::{Deserialize, Serialize};

/// VAE configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
        }
    }
}

/// Resnet block for VAE.
#[derive(Debug)]
pub struct ResnetBlock {
    pub norm1: nn::GroupNorm,
    pub conv1: nn::Conv2d,
    pub norm2: nn::GroupNorm,
    pub conv2: nn::Conv2d,
    pub conv_shortcut: Option<nn::Conv2d>,
}
impl_module_params!(ResnetBlock; norm1, conv1, norm2, conv2, conv_shortcut);

impl ResnetBlock {
    pub fn new(in_channels: usize, out_channels: usize, groups: usize, eps: f32) -> Self {
        let norm1 = nn::GroupNormBuilder::new(groups as i32, in_channels as i32)
            .eps(eps)
            .build()
            .unwrap();
        let conv1 = nn::Conv2dBuilder::new(in_channels as i32, out_channels as i32, 3)
            .padding(1)
            .build()
            .unwrap();
        let norm2 = nn::GroupNormBuilder::new(groups as i32, out_channels as i32)
            .eps(eps)
            .build()
            .unwrap();
        let conv2 = nn::Conv2dBuilder::new(out_channels as i32, out_channels as i32, 3)
            .padding(1)
            .build()
            .unwrap();

        let conv_shortcut = if in_channels != out_channels {
            Some(
                nn::Conv2dBuilder::new(in_channels as i32, out_channels as i32, 1)
                    .build()
                    .unwrap(),
            )
        } else {
            None
        };

        Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.norm1.forward(x);
        h = nn::silu(&h);
        h = self.conv1.forward(&h);
        h = self.norm2.forward(&h);
        h = nn::silu(&h);
        h = self.conv2.forward(&h);

        let residual = if let Some(ref mut shortcut) = self.conv_shortcut {
            shortcut.forward(x)
        } else {
            x.clone()
        };

        Ok(h.add(&residual))
    }
}

/// UpSampler block for VAE.
#[derive(Debug)]
pub struct UpSampler {
    pub conv: nn::Conv2d,
}
impl_module_params!(UpSampler; conv);

impl UpSampler {
    pub fn new(channels: usize) -> Self {
        let conv = nn::Conv2dBuilder::new(channels as i32, channels as i32, 3)
            .padding(1)
            .build()
            .unwrap();
        Self { conv }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let b = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        // NHWC upsampling: [B, H, 1, W, 1, C] -> [B, H, 2, W, 2, C] -> [B, H*2, W*2, C]
        let x = x.expand_dims_axes(&[2, 4]);
        let x = pmetal_bridge::compat::ops::broadcast_to(&x, &[b, h, 2, w, 2, c]);
        let x = x.reshape(&[b, h * 2, w * 2, c]);

        Ok(self.conv.forward(&x))
    }
}

/// DownSampler block for VAE.
#[derive(Debug)]
pub struct DownSampler {
    pub conv: nn::Conv2d,
}
impl_module_params!(DownSampler; conv);

impl DownSampler {
    pub fn new(channels: usize) -> Self {
        let conv = nn::Conv2dBuilder::new(channels as i32, channels as i32, 3)
            .stride(2)
            .padding(1)
            .build()
            .unwrap();
        Self { conv }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        Ok(self.conv.forward(x))
    }
}

/// Attention block for VAE.
#[derive(Debug)]
pub struct VAEAttentionBlock {
    pub norm: nn::GroupNorm,
    pub q: nn::Conv2d,
    pub k: nn::Conv2d,
    pub v: nn::Conv2d,
    pub proj_out: nn::Conv2d,
}
impl_module_params!(VAEAttentionBlock; norm, q, k, v, proj_out);

impl VAEAttentionBlock {
    pub fn new(channels: usize, groups: usize, eps: f32) -> Self {
        let norm = nn::GroupNormBuilder::new(groups as i32, channels as i32)
            .eps(eps)
            .build()
            .unwrap();
        let q = nn::Conv2dBuilder::new(channels as i32, channels as i32, 1)
            .build()
            .unwrap();
        let k = nn::Conv2dBuilder::new(channels as i32, channels as i32, 1)
            .build()
            .unwrap();
        let v = nn::Conv2dBuilder::new(channels as i32, channels as i32, 1)
            .build()
            .unwrap();
        let proj_out = nn::Conv2dBuilder::new(channels as i32, channels as i32, 1)
            .build()
            .unwrap();

        Self {
            norm,
            q,
            k,
            v,
            proj_out,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let b = x.dim(0);
        let h = x.dim(1);
        let w = x.dim(2);
        let c = x.dim(3);

        let residual = x;
        let h_norm = self.norm.forward(x);

        let q = self.q.forward(&h_norm);
        let k = self.k.forward(&h_norm);
        let v = self.v.forward(&h_norm);

        // NHWC attention: flatten H*W into sequence dimension
        let q = q.reshape(&[b, h * w, c]).expand_dims_axes(&[1]);
        let k = k.reshape(&[b, h * w, c]).expand_dims_axes(&[1]);
        let v = v.reshape(&[b, h * w, c]).expand_dims_axes(&[1]);

        let scale = 1.0 / (c as f32).sqrt();
        let attn_out =
            pmetal_bridge::compat::fast::scaled_dot_product_attention(&q, &k, &v, scale, "none");

        let attn_out = attn_out.squeeze_axes(&[1]).reshape(&[b, h, w, c]);
        let out = self.proj_out.forward(&attn_out);

        Ok(residual.add(&out))
    }
}

/// Flux VAE Encoder.
#[derive(Debug)]
pub struct FluxVAEEncoder {
    pub conv_in: nn::Conv2d,

    pub down_1_0: ResnetBlock,
    pub down_1_1: ResnetBlock,

    pub down_2_0: ResnetBlock,
    pub down_2_1: ResnetBlock,
    pub down_2_sampler: DownSampler,

    pub down_3_0: ResnetBlock,
    pub down_3_1: ResnetBlock,
    pub down_3_sampler: DownSampler,

    pub down_4_0: ResnetBlock,
    pub down_4_1: ResnetBlock,
    pub down_4_sampler: DownSampler,

    pub mid_block_1: ResnetBlock,
    pub mid_attn: VAEAttentionBlock,
    pub mid_block_2: ResnetBlock,

    pub norm_out: nn::GroupNorm,
    pub conv_out: nn::Conv2d,
}
impl_module_params!(FluxVAEEncoder; conv_in, down_1_0, down_1_1, down_2_0, down_2_1, down_2_sampler, down_3_0, down_3_1, down_3_sampler, down_4_0, down_4_1, down_4_sampler, mid_block_1, mid_attn, mid_block_2, norm_out, conv_out);

impl Default for FluxVAEEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FluxVAEEncoder {
    pub fn new() -> Self {
        let conv_in = nn::Conv2dBuilder::new(3, 128, 3)
            .padding(1)
            .build()
            .unwrap();

        let down_1_0 = ResnetBlock::new(128, 128, 32, 1e-6);
        let down_1_1 = ResnetBlock::new(128, 128, 32, 1e-6);

        let down_2_sampler = DownSampler::new(128);
        let down_2_0 = ResnetBlock::new(128, 256, 32, 1e-6);
        let down_2_1 = ResnetBlock::new(256, 256, 32, 1e-6);

        let down_3_sampler = DownSampler::new(256);
        let down_3_0 = ResnetBlock::new(256, 512, 32, 1e-6);
        let down_3_1 = ResnetBlock::new(512, 512, 32, 1e-6);

        let down_4_sampler = DownSampler::new(512);
        let down_4_0 = ResnetBlock::new(512, 512, 32, 1e-6);
        let down_4_1 = ResnetBlock::new(512, 512, 32, 1e-6);

        let mid_block_1 = ResnetBlock::new(512, 512, 32, 1e-6);
        let mid_attn = VAEAttentionBlock::new(512, 32, 1e-6);
        let mid_block_2 = ResnetBlock::new(512, 512, 32, 1e-6);

        let norm_out = nn::GroupNormBuilder::new(32, 512)
            .eps(1e-6)
            .build()
            .unwrap();
        let conv_out = nn::Conv2dBuilder::new(512, 32, 3)
            .padding(1)
            .build()
            .unwrap();

        Self {
            conv_in,
            down_1_0,
            down_1_1,
            down_2_0,
            down_2_1,
            down_2_sampler,
            down_3_0,
            down_3_1,
            down_3_sampler,
            down_4_0,
            down_4_1,
            down_4_sampler,
            mid_block_1,
            mid_attn,
            mid_block_2,
            norm_out,
            conv_out,
        }
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.conv_in.forward(x);

        h = self.down_1_0.forward(&h)?;
        h = self.down_1_1.forward(&h)?;

        h = self.down_2_sampler.forward(&h)?;
        h = self.down_2_0.forward(&h)?;
        h = self.down_2_1.forward(&h)?;

        h = self.down_3_sampler.forward(&h)?;
        h = self.down_3_0.forward(&h)?;
        h = self.down_3_1.forward(&h)?;

        h = self.down_4_sampler.forward(&h)?;
        h = self.down_4_0.forward(&h)?;
        h = self.down_4_1.forward(&h)?;

        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        h = self.norm_out.forward(&h);
        h = nn::silu(&h);
        h = self.conv_out.forward(&h);

        Ok(h)
    }
}

/// Flux VAE Decoder.
#[derive(Debug)]
pub struct FluxVAEDecoder {
    pub conv_in: nn::Conv2d,
    pub mid_block_1: ResnetBlock,
    pub mid_attn: VAEAttentionBlock,
    pub mid_block_2: ResnetBlock,

    pub up_1_0: ResnetBlock,
    pub up_1_1: ResnetBlock,
    pub up_1_2: ResnetBlock,
    pub up_1_sampler: UpSampler,

    pub up_2_0: ResnetBlock,
    pub up_2_1: ResnetBlock,
    pub up_2_2: ResnetBlock,
    pub up_2_sampler: UpSampler,

    pub up_3_0: ResnetBlock,
    pub up_3_1: ResnetBlock,
    pub up_3_2: ResnetBlock,
    pub up_3_sampler: UpSampler,

    pub up_4_0: ResnetBlock,
    pub up_4_1: ResnetBlock,
    pub up_4_2: ResnetBlock,

    pub norm_out: nn::GroupNorm,
    pub conv_out: nn::Conv2d,
}
impl_module_params!(FluxVAEDecoder; conv_in, mid_block_1, mid_attn, mid_block_2, up_1_0, up_1_1, up_1_2, up_1_sampler, up_2_0, up_2_1, up_2_2, up_2_sampler, up_3_0, up_3_1, up_3_2, up_3_sampler, up_4_0, up_4_1, up_4_2, norm_out, conv_out);

impl Default for FluxVAEDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl FluxVAEDecoder {
    pub fn new() -> Self {
        let conv_in = nn::Conv2dBuilder::new(16, 512, 3)
            .padding(1)
            .build()
            .unwrap();

        let mid_block_1 = ResnetBlock::new(512, 512, 32, 1e-6);
        let mid_attn = VAEAttentionBlock::new(512, 32, 1e-6);
        let mid_block_2 = ResnetBlock::new(512, 512, 32, 1e-6);

        let up_1_0 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_1_1 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_1_2 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_1_sampler = UpSampler::new(512);

        let up_2_0 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_2_1 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_2_2 = ResnetBlock::new(512, 512, 32, 1e-6);
        let up_2_sampler = UpSampler::new(512);

        let up_3_0 = ResnetBlock::new(512, 256, 32, 1e-6);
        let up_3_1 = ResnetBlock::new(256, 256, 32, 1e-6);
        let up_3_2 = ResnetBlock::new(256, 256, 32, 1e-6);
        let up_3_sampler = UpSampler::new(256);

        let up_4_0 = ResnetBlock::new(256, 128, 32, 1e-6);
        let up_4_1 = ResnetBlock::new(128, 128, 32, 1e-6);
        let up_4_2 = ResnetBlock::new(128, 128, 32, 1e-6);

        let norm_out = nn::GroupNormBuilder::new(32, 128)
            .eps(1e-6)
            .build()
            .unwrap();
        let conv_out = nn::Conv2dBuilder::new(128, 3, 3)
            .padding(1)
            .build()
            .unwrap();

        Self {
            conv_in,
            mid_block_1,
            mid_attn,
            mid_block_2,
            up_1_0,
            up_1_1,
            up_1_2,
            up_1_sampler,
            up_2_0,
            up_2_1,
            up_2_2,
            up_2_sampler,
            up_3_0,
            up_3_1,
            up_3_2,
            up_3_sampler,
            up_4_0,
            up_4_1,
            up_4_2,
            norm_out,
            conv_out,
        }
    }

    pub fn forward(&mut self, z: &Array) -> Result<Array, Exception> {
        let mut h = self.conv_in.forward(z);

        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        h = self.up_1_0.forward(&h)?;
        h = self.up_1_1.forward(&h)?;
        h = self.up_1_2.forward(&h)?;
        h = self.up_1_sampler.forward(&h)?;

        h = self.up_2_0.forward(&h)?;
        h = self.up_2_1.forward(&h)?;
        h = self.up_2_2.forward(&h)?;
        h = self.up_2_sampler.forward(&h)?;

        h = self.up_3_0.forward(&h)?;
        h = self.up_3_1.forward(&h)?;
        h = self.up_3_2.forward(&h)?;
        h = self.up_3_sampler.forward(&h)?;

        h = self.up_4_0.forward(&h)?;
        h = self.up_4_1.forward(&h)?;
        h = self.up_4_2.forward(&h)?;

        h = self.norm_out.forward(&h);
        h = nn::silu(&h);
        h = self.conv_out.forward(&h);

        Ok(h)
    }
}

/// Flux VAE model.
#[derive(Debug)]
pub struct FluxVAE {
    pub encoder: Option<FluxVAEEncoder>,
    pub decoder: FluxVAEDecoder,
}
impl_module_params!(FluxVAE; encoder, decoder);

impl FluxVAE {
    pub const SCALING_FACTOR: f32 = 0.3611;
    pub const SHIFT_FACTOR: f32 = 0.1159;

    pub fn new(_config: VAEConfig) -> Self {
        Self {
            encoder: Some(FluxVAEEncoder::new()),
            decoder: FluxVAEDecoder::new(),
        }
    }

    /// Encode an image into latents.
    ///
    /// # Arguments
    /// * `x` - Input image [B, H, W, 3] in range [-1, 1]
    /// * `sample` - Whether to sample from the distribution or return the mean
    pub fn encode(&mut self, x: &Array, sample: bool) -> Result<Array, Exception> {
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| Exception::custom("Encoder not initialized"))?;
        let h = encoder.forward(x)?;

        // Output is 32 channels: [mean (16), logvar (16)]
        let mean = pmetal_bridge::compat::ops::slice_last_to(&h, 16);

        if sample {
            let logvar = pmetal_bridge::compat::ops::slice_last_from(&h, 16);
            // Clamp logvar to prevent numerical overflow in exp()
            let logvar = pmetal_bridge::compat::ops::clip(
                &logvar,
                Some(&Array::from_f32(-30.0)),
                Some(&Array::from_f32(20.0)),
            );
            let std = logvar.multiply(&Array::from_f32(0.5)).exp();
            let noise = pmetal_bridge::compat::random::normal(mean.shape(), mean.dtype());
            let z = mean.add(&std.multiply(&noise));
            Ok(z.subtract(&Array::from_f32(Self::SHIFT_FACTOR))
                .multiply(&Array::from_f32(Self::SCALING_FACTOR)))
        } else {
            Ok(mean
                .subtract(&Array::from_f32(Self::SHIFT_FACTOR))
                .multiply(&Array::from_f32(Self::SCALING_FACTOR)))
        }
    }

    pub fn decode(&mut self, z: &Array) -> Result<Array, Exception> {
        let z = z
            .divide(&Array::from_f32(Self::SCALING_FACTOR))
            .add(&Array::from_f32(Self::SHIFT_FACTOR));
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_bridge::compat::ops::zeros;

    #[test]
    fn test_flux_vae_roundtrip() {
        let config = VAEConfig::default();
        let mut vae = FluxVAE::new(config);

        // Dummy input image: [1, 64, 64, 3]
        let x = zeros(&[1, 64, 64, 3], pmetal_bridge::compat::Dtype::Float32);

        // Encode
        let z = vae.encode(&x, false).unwrap();
        assert_eq!(z.shape(), &[1, 8, 8, 16]); // 64/8 = 8

        // Decode
        let x_hat = vae.decode(&z).unwrap();
        assert_eq!(x_hat.shape(), &[1, 64, 64, 3]);
    }
}
