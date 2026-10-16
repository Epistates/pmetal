//! `AutoencoderKL`: the variational autoencoder of latent diffusion.
//!
//! The KL-regularised convolutional VAE of Rombach et al. (2022), as
//! `diffusers`' `AutoencoderKL` defines it. FLUX, SD3 and Stable Diffusion
//! differ only in configuration (latent width, whether the two 1x1
//! `quant_conv`s exist, the latent scale and shift), so one implementation
//! builds all of them from `config.json`.
//!
//! Images and latents are **channels-last** (`[batch, height, width,
//! channels]`), MLX's layout; checkpoints are channels-first and the loader
//! transposes each convolution weight once.
//!
//! The parameter tree reuses `diffusers`' own names (`encoder.down_blocks.0.
//! resnets.1.conv1`, `decoder.mid_block.attentions.0.to_q`, ...), so loading
//! needs no name table.
//!
//! The mid-block's single-head self-attention runs over every pixel, so at
//! megapixel sizes its score matrix, not the activations, dominates memory;
//! [`VAEAttentionBlock::forward`] walks the queries in chunks (each row of
//! scores is independent) to bound it.

use pmetal_runtime::compat::{Array, Exception, nn, ops};
use pmetal_runtime::impl_module_params;
use pmetal_runtime::nn::{AttentionMaskType, FusedAttentionConfig, fused_sdpa};
use serde::{Deserialize, Serialize};

/// `AutoencoderKL` configuration, with the defaults of its `diffusers` config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    #[serde(default = "default_image_channels")]
    pub in_channels: usize,
    #[serde(default = "default_image_channels")]
    pub out_channels: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    /// Channels of each resolution level, highest resolution first. There is
    /// one downsampling between consecutive levels.
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize,
    /// Whether the encoder's moments pass through a 1x1 `quant_conv` (SD yes,
    /// FLUX no).
    #[serde(default = "default_true")]
    pub use_quant_conv: bool,
    /// Whether the latents pass through a 1x1 `post_quant_conv` before the
    /// decoder (SD yes, FLUX no).
    #[serde(default = "default_true")]
    pub use_post_quant_conv: bool,
    /// Whether the mid-blocks carry their self-attention.
    #[serde(default = "default_true")]
    pub mid_block_add_attention: bool,
    /// Latents are multiplied by this after encoding and divided by it before
    /// decoding, to bring them to unit variance.
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f32,
    /// Subtracted before scaling when encoding, added back after unscaling
    /// when decoding (FLUX, SD3). `None` is no shift.
    #[serde(default)]
    pub shift_factor: Option<f32>,
}

fn default_image_channels() -> usize {
    3
}
fn default_latent_channels() -> usize {
    4
}
fn default_block_out_channels() -> Vec<usize> {
    vec![64]
}
fn default_layers_per_block() -> usize {
    1
}
fn default_norm_num_groups() -> usize {
    32
}
fn default_true() -> bool {
    true
}
fn default_scaling_factor() -> f32 {
    0.18215
}

impl Default for VAEConfig {
    /// `diffusers`' `AutoencoderKL()` defaults.
    fn default() -> Self {
        Self {
            in_channels: default_image_channels(),
            out_channels: default_image_channels(),
            latent_channels: default_latent_channels(),
            block_out_channels: default_block_out_channels(),
            layers_per_block: default_layers_per_block(),
            norm_num_groups: default_norm_num_groups(),
            use_quant_conv: true,
            use_post_quant_conv: true,
            mid_block_add_attention: true,
            scaling_factor: default_scaling_factor(),
            shift_factor: None,
        }
    }
}

impl VAEConfig {
    /// The 16-channel VAE of FLUX.1: four levels, no quant convs, scale 0.3611
    /// and shift 0.1159.
    pub fn flux() -> Self {
        Self {
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            use_quant_conv: false,
            use_post_quant_conv: false,
            scaling_factor: 0.3611,
            shift_factor: Some(0.1159),
            ..Self::default()
        }
    }

    /// Reject the configurations the blocks cannot be built from.
    fn validate(&self) -> Result<(), Exception> {
        if self.block_out_channels.is_empty() {
            return Err(Exception::custom("vae: block_out_channels is empty"));
        }
        let groups = self.norm_num_groups;
        for &channels in &self.block_out_channels {
            if groups == 0 || channels % groups != 0 {
                return Err(Exception::custom(format!(
                    "vae: {channels} channels do not divide into {groups} normalisation groups"
                )));
            }
        }
        Ok(())
    }
}

const NORM_EPS: f32 = 1e-6;

fn group_norm(groups: usize, channels: usize) -> Result<nn::GroupNorm, Exception> {
    nn::GroupNormBuilder::new(groups as i32, channels as i32)
        .eps(NORM_EPS)
        .build()
}

fn conv(
    inputs: usize,
    outputs: usize,
    kernel: i32,
    stride: i32,
    padding: i32,
) -> Result<nn::Conv2d, Exception> {
    nn::Conv2dBuilder::new(inputs as i32, outputs as i32, kernel)
        .stride(stride)
        .padding(padding)
        .build()
}

/// Resnet block: two (norm, SiLU, 3x3 conv) stages and a skip connection,
/// through a 1x1 conv when the width changes.
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
    pub fn new(inputs: usize, outputs: usize, groups: usize) -> Result<Self, Exception> {
        Ok(Self {
            norm1: group_norm(groups, inputs)?,
            conv1: conv(inputs, outputs, 3, 1, 1)?,
            norm2: group_norm(groups, outputs)?,
            conv2: conv(outputs, outputs, 3, 1, 1)?,
            conv_shortcut: (inputs != outputs)
                .then(|| conv(inputs, outputs, 1, 1, 0))
                .transpose()?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.conv1.forward(&nn::silu(&self.norm1.forward(x)));
        let h = self.conv2.forward(&nn::silu(&self.norm2.forward(&h)));
        let skip = match self.conv_shortcut.as_mut() {
            Some(shortcut) => shortcut.forward(x),
            None => x.clone(),
        };
        Ok(skip.add(&h))
    }
}

/// Nearest-neighbour 2x upsampling followed by a 3x3 conv.
#[derive(Debug)]
pub struct UpSampler {
    pub conv: nn::Conv2d,
}
impl_module_params!(UpSampler; conv);

impl UpSampler {
    pub fn new(channels: usize) -> Result<Self, Exception> {
        Ok(Self {
            conv: conv(channels, channels, 3, 1, 1)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        // [B, H, 1, W, 1, C] -> [B, H, 2, W, 2, C] -> [B, 2H, 2W, C]
        let x = x.expand_dims_axes(&[2, 4]);
        let x = ops::broadcast_to(&x, &[b, h, 2, w, 2, c]).reshape(&[b, h * 2, w * 2, c]);
        Ok(self.conv.forward(&x))
    }
}

/// Stride-2 3x3 conv that pads one pixel on the bottom and right only, as
/// `diffusers`' `Downsample2D(padding=0)` does. A symmetric padding of one
/// samples the same stride-2 grid shifted by a pixel, which is a different
/// image.
#[derive(Debug)]
pub struct DownSampler {
    pub conv: nn::Conv2d,
}
impl_module_params!(DownSampler; conv);

impl DownSampler {
    pub fn new(channels: usize) -> Result<Self, Exception> {
        Ok(Self {
            conv: conv(channels, channels, 3, 2, 0)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let padded = ops::pad(x, &[(0, 0), (0, 1), (0, 1), (0, 0)], None, None);
        Ok(self.conv.forward(&padded))
    }
}

/// The scores one chunk of queries may form, in elements (256 MiB in f32).
const ATTENTION_CHUNK_ELEMENTS: i64 = 1 << 26;

/// The mid-block's single-head self-attention over every pixel.
#[derive(Debug)]
pub struct VAEAttentionBlock {
    pub group_norm: nn::GroupNorm,
    pub to_q: nn::Linear,
    pub to_k: nn::Linear,
    pub to_v: nn::Linear,
    /// `diffusers` keeps the output projection (and its dropout) in a list.
    pub to_out: Vec<nn::Linear>,
}
impl_module_params!(VAEAttentionBlock; group_norm, to_q, to_k, to_v, to_out);

impl VAEAttentionBlock {
    pub fn new(channels: usize, groups: usize) -> Result<Self, Exception> {
        let linear = || nn::LinearBuilder::new(channels as i32, channels as i32).build();
        Ok(Self {
            group_norm: group_norm(groups, channels)?,
            to_q: linear()?,
            to_k: linear()?,
            to_v: linear()?,
            to_out: vec![linear()?],
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let tokens = x.dim(1) * x.dim(2);
        let rows = (ATTENTION_CHUNK_ELEMENTS / i64::from(tokens)).max(1) as i32;
        self.forward_in_chunks(x, rows)
    }

    /// [`forward`](Self::forward) with at most `rows` queries scored against
    /// the keys at a time.
    pub fn forward_in_chunks(&mut self, x: &Array, rows: i32) -> Result<Array, Exception> {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3));
        let tokens = h * w;
        let normed = self.group_norm.forward(x).reshape(&[b, tokens, c]);
        let head = |t: Array| t.reshape(&[b, 1, tokens, c]);
        let q = head(self.to_q.forward(&normed));
        let k = head(self.to_k.forward(&normed));
        let v = head(self.to_v.forward(&normed));

        let config = FusedAttentionConfig::new(1, 1, c).with_mask_type(AttentionMaskType::None);
        let out = if rows >= tokens {
            fused_sdpa(&q, &k, &v, &config, None)?
        } else {
            // Every query row's softmax is its own, so walking the queries in
            // chunks gives the whole attention's rows exactly, with scores of
            // `rows x tokens` at a time.
            let mut chunks = Vec::with_capacity((tokens / rows + 1) as usize);
            let mut start = 0;
            while start < tokens {
                let end = (start + rows).min(tokens);
                let queries = ops::slice_axis(&q, 2, start, end);
                chunks.push(fused_sdpa(&queries, &k, &v, &config, None)?);
                start = end;
            }
            ops::concatenate_axis(&chunks.iter().collect::<Vec<_>>(), 2)
        };

        let out = out.reshape(&[b, tokens, c]);
        let out = self
            .to_out
            .first_mut()
            .ok_or_else(|| Exception::custom("vae: attention has no output projection"))?
            .forward(&out)
            .reshape(&[b, h, w, c]);
        Ok(x.add(&out))
    }
}

/// resnet, optional attention, resnet.
#[derive(Debug)]
pub struct MidBlock {
    pub resnets: Vec<ResnetBlock>,
    pub attentions: Vec<VAEAttentionBlock>,
}
impl_module_params!(MidBlock; resnets, attentions);

impl MidBlock {
    pub fn new(channels: usize, groups: usize, attention: bool) -> Result<Self, Exception> {
        Ok(Self {
            resnets: vec![
                ResnetBlock::new(channels, channels, groups)?,
                ResnetBlock::new(channels, channels, groups)?,
            ],
            attentions: if attention {
                vec![VAEAttentionBlock::new(channels, groups)?]
            } else {
                Vec::new()
            },
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.resnets[0].forward(x)?;
        if let Some(attention) = self.attentions.first_mut() {
            h = attention.forward(&h)?;
        }
        self.resnets[1].forward(&h)
    }
}

/// A resolution level of the encoder: resnets, then a downsampler on all but
/// the last level.
#[derive(Debug)]
pub struct DownBlock {
    pub resnets: Vec<ResnetBlock>,
    pub downsamplers: Vec<DownSampler>,
}
impl_module_params!(DownBlock; resnets, downsamplers);

impl DownBlock {
    fn new(
        inputs: usize,
        outputs: usize,
        layers: usize,
        groups: usize,
        downsample: bool,
    ) -> Result<Self, Exception> {
        let resnets = (0..layers)
            .map(|i| ResnetBlock::new(if i == 0 { inputs } else { outputs }, outputs, groups))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            resnets,
            downsamplers: if downsample {
                vec![DownSampler::new(outputs)?]
            } else {
                Vec::new()
            },
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for resnet in &mut self.resnets {
            h = resnet.forward(&h)?;
        }
        for downsampler in &mut self.downsamplers {
            h = downsampler.forward(&h)?;
        }
        Ok(h)
    }
}

/// A resolution level of the decoder: resnets, then an upsampler on all but
/// the last level.
#[derive(Debug)]
pub struct UpBlock {
    pub resnets: Vec<ResnetBlock>,
    pub upsamplers: Vec<UpSampler>,
}
impl_module_params!(UpBlock; resnets, upsamplers);

impl UpBlock {
    fn new(
        inputs: usize,
        outputs: usize,
        layers: usize,
        groups: usize,
        upsample: bool,
    ) -> Result<Self, Exception> {
        let resnets = (0..layers)
            .map(|i| ResnetBlock::new(if i == 0 { inputs } else { outputs }, outputs, groups))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            resnets,
            upsamplers: if upsample {
                vec![UpSampler::new(outputs)?]
            } else {
                Vec::new()
            },
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for resnet in &mut self.resnets {
            h = resnet.forward(&h)?;
        }
        for upsampler in &mut self.upsamplers {
            h = upsampler.forward(&h)?;
        }
        Ok(h)
    }
}

/// Image to moments (`2 * latent_channels` channels: mean, then log-variance).
#[derive(Debug)]
pub struct VAEEncoder {
    pub conv_in: nn::Conv2d,
    pub down_blocks: Vec<DownBlock>,
    pub mid_block: MidBlock,
    pub conv_norm_out: nn::GroupNorm,
    pub conv_out: nn::Conv2d,
}
impl_module_params!(VAEEncoder; conv_in, down_blocks, mid_block, conv_norm_out, conv_out);

impl VAEEncoder {
    pub fn new(config: &VAEConfig) -> Result<Self, Exception> {
        let widths = &config.block_out_channels;
        let groups = config.norm_num_groups;
        let mut previous = widths[0];
        let mut down_blocks = Vec::with_capacity(widths.len());
        for (i, &width) in widths.iter().enumerate() {
            down_blocks.push(DownBlock::new(
                previous,
                width,
                config.layers_per_block,
                groups,
                i + 1 < widths.len(),
            )?);
            previous = width;
        }
        let last = widths[widths.len() - 1];
        Ok(Self {
            conv_in: conv(config.in_channels, widths[0], 3, 1, 1)?,
            down_blocks,
            mid_block: MidBlock::new(last, groups, config.mid_block_add_attention)?,
            conv_norm_out: group_norm(groups, last)?,
            conv_out: conv(last, 2 * config.latent_channels, 3, 1, 1)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.conv_in.forward(x);
        for block in &mut self.down_blocks {
            h = block.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        Ok(self
            .conv_out
            .forward(&nn::silu(&self.conv_norm_out.forward(&h))))
    }
}

/// Latents to image.
#[derive(Debug)]
pub struct VAEDecoder {
    pub conv_in: nn::Conv2d,
    pub mid_block: MidBlock,
    pub up_blocks: Vec<UpBlock>,
    pub conv_norm_out: nn::GroupNorm,
    pub conv_out: nn::Conv2d,
}
impl_module_params!(VAEDecoder; conv_in, mid_block, up_blocks, conv_norm_out, conv_out);

impl VAEDecoder {
    pub fn new(config: &VAEConfig) -> Result<Self, Exception> {
        let widths: Vec<usize> = config.block_out_channels.iter().rev().copied().collect();
        let groups = config.norm_num_groups;
        let mut previous = widths[0];
        let mut up_blocks = Vec::with_capacity(widths.len());
        for (i, &width) in widths.iter().enumerate() {
            up_blocks.push(UpBlock::new(
                previous,
                width,
                config.layers_per_block + 1,
                groups,
                i + 1 < widths.len(),
            )?);
            previous = width;
        }
        let first = config.block_out_channels[0];
        Ok(Self {
            conv_in: conv(config.latent_channels, widths[0], 3, 1, 1)?,
            mid_block: MidBlock::new(widths[0], groups, config.mid_block_add_attention)?,
            up_blocks,
            conv_norm_out: group_norm(groups, first)?,
            conv_out: conv(first, config.out_channels, 3, 1, 1)?,
        })
    }

    pub fn forward(&mut self, z: &Array) -> Result<Array, Exception> {
        let mut h = self.mid_block.forward(&self.conv_in.forward(z))?;
        for block in &mut self.up_blocks {
            h = block.forward(&h)?;
        }
        Ok(self
            .conv_out
            .forward(&nn::silu(&self.conv_norm_out.forward(&h))))
    }
}

/// The mean and log-variance of the latent distribution an image encodes to.
#[derive(Debug, Clone)]
pub struct LatentDistribution {
    /// `[batch, height / 2^(levels-1), width / 2^(levels-1), latent_channels]`.
    pub mean: Array,
    /// Clamped to `[-30, 20]` as `DiagonalGaussianDistribution` does.
    pub logvar: Array,
}

impl LatentDistribution {
    /// A draw `mean + exp(logvar / 2) * eps`, `eps ~ N(0, 1)`.
    pub fn sample(&self) -> Array {
        let std = self.logvar.mul_scalar(0.5).exp();
        let noise = pmetal_runtime::compat::random::normal(self.mean.shape(), self.mean.dtype());
        self.mean.add(&std.multiply(&noise))
    }
}

/// `AutoencoderKL`.
#[derive(Debug)]
pub struct AutoencoderKL {
    pub encoder: VAEEncoder,
    pub decoder: VAEDecoder,
    pub quant_conv: Option<nn::Conv2d>,
    pub post_quant_conv: Option<nn::Conv2d>,
    /// Model configuration (not a parameter).
    pub config: VAEConfig,
}
impl_module_params!(AutoencoderKL; encoder, decoder, quant_conv, post_quant_conv);

impl AutoencoderKL {
    pub fn new(config: VAEConfig) -> Result<Self, Exception> {
        config.validate()?;
        let latent = config.latent_channels;
        Ok(Self {
            encoder: VAEEncoder::new(&config)?,
            decoder: VAEDecoder::new(&config)?,
            quant_conv: config
                .use_quant_conv
                .then(|| conv(2 * latent, 2 * latent, 1, 1, 0))
                .transpose()?,
            post_quant_conv: config
                .use_post_quant_conv
                .then(|| conv(latent, latent, 1, 1, 0))
                .transpose()?,
            config,
        })
    }

    pub fn config(&self) -> &VAEConfig {
        &self.config
    }

    /// The latent distribution of an image batch `x` (`[batch, height, width,
    /// in_channels]`, in `[-1, 1]`), before any scaling.
    pub fn encode_distribution(&mut self, x: &Array) -> Result<LatentDistribution, Exception> {
        let mut moments = self.encoder.forward(x)?;
        if let Some(quant_conv) = self.quant_conv.as_mut() {
            moments = quant_conv.forward(&moments);
        }
        let channels = self.config.latent_channels as i32;
        let mean = ops::slice_last_to(&moments, channels);
        let logvar = ops::slice_last_from(&moments, channels);
        let logvar = ops::clip(
            &logvar,
            Some(&Array::scalar_like(-30.0, &logvar)),
            Some(&Array::scalar_like(20.0, &logvar)),
        );
        Ok(LatentDistribution { mean, logvar })
    }

    /// Encode an image batch to the latents a diffusion model works in:
    /// `(z - shift_factor) * scaling_factor`, with `z` the distribution's mean
    /// or, if `sample`, a draw from it.
    pub fn encode(&mut self, x: &Array, sample: bool) -> Result<Array, Exception> {
        let distribution = self.encode_distribution(x)?;
        let z = if sample {
            distribution.sample()
        } else {
            distribution.mean
        };
        Ok(self.scale_latents(&z))
    }

    /// `(z - shift_factor) * scaling_factor`.
    pub fn scale_latents(&self, z: &Array) -> Array {
        let shifted = match self.config.shift_factor {
            Some(shift) => z.sub_scalar(shift),
            None => z.clone(),
        };
        shifted.mul_scalar(self.config.scaling_factor)
    }

    /// The inverse of [`scale_latents`](Self::scale_latents):
    /// `z / scaling_factor + shift_factor`.
    pub fn unscale_latents(&self, z: &Array) -> Array {
        let unscaled = z.div_scalar(self.config.scaling_factor);
        match self.config.shift_factor {
            Some(shift) => unscaled.add_scalar(shift),
            None => unscaled,
        }
    }

    /// Decode raw (unscaled) latents to an image batch.
    pub fn decode_raw(&mut self, z: &Array) -> Result<Array, Exception> {
        let z = match self.post_quant_conv.as_mut() {
            Some(post_quant_conv) => post_quant_conv.forward(z),
            None => z.clone(),
        };
        self.decoder.forward(&z)
    }

    /// Decode the latents of a diffusion model (see
    /// [`encode`](Self::encode)) to an image batch in `[-1, 1]`.
    pub fn decode(&mut self, z: &Array) -> Result<Array, Exception> {
        let raw = self.unscale_latents(z);
        self.decode_raw(&raw)
    }
}
