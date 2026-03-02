//! Flux.1 inference pipeline.
//!
//! Coordinates CLIP/T5 text encoders, Flux DiT, and VAE for end-to-end
//! high-quality image generation on Apple Silicon.

use mlx_rs::{Array, error::Exception, module::Module};
use pmetal_core::Result;
use std::path::Path;

use crate::architectures::clip::{CLIPConfig, CLIPTextModel};
use crate::architectures::flux::{FluxConfig, FluxDiT};
use crate::architectures::t5::{T5Config, T5EncoderModel};
use crate::architectures::vae::{FluxVAE, VAEConfig};
use crate::sampling::diffusion::FlowMatchScheduler;

/// Flux.1 Pipeline for text-to-image generation.
pub struct FluxPipeline {
    pub clip: CLIPTextModel,
    pub t5: T5EncoderModel,
    pub dit: FluxDiT,
    pub vae: FluxVAE,
    pub scheduler: FlowMatchScheduler,
}

impl FluxPipeline {
    /// Create a new Flux.1 pipeline from configuration.
    pub fn new(
        clip_config: CLIPConfig,
        t5_config: T5Config,
        flux_config: FluxConfig,
        vae_config: VAEConfig,
    ) -> Self {
        let clip = CLIPTextModel::new(clip_config);
        let t5 = T5EncoderModel::new(t5_config);
        let dit = FluxDiT::new(flux_config);
        let vae = FluxVAE::new(vae_config);
        let scheduler = FlowMatchScheduler::new_flux(20, 1.0, Some(3.0)).unwrap();

        Self {
            clip,
            t5,
            dit,
            vae,
            scheduler,
        }
    }

    /// Load the pipeline from a model directory.
    pub fn from_pretrained(_model_dir: impl AsRef<Path>) -> Result<Self> {
        Err(pmetal_core::PMetalError::Config(
            "FluxPipeline::from_pretrained is not yet implemented. \
             Use FluxPipeline::new() with explicit configs and load weights manually."
                .to_string(),
        ))
    }

    /// Generate an image from a prompt.
    ///
    /// # Arguments
    /// * `clip_input` - Tokenized prompt for CLIP (usually 77 tokens).
    /// * `t5_input` - Tokenized prompt for T5 (usually up to 256 or 512 tokens).
    /// * `width` - Output image width.
    /// * `height` - Output image height.
    /// * `num_steps` - Number of denoising steps.
    /// * `guidance` - Guidance scale for Flux.
    /// * `seed` - Optional random seed.
    pub fn generate(
        &mut self,
        clip_input: &Array,
        t5_input: &Array,
        width: usize,
        height: usize,
        num_steps: usize,
        guidance: f32,
        seed: Option<u64>,
    ) -> Result<Array> {
        let batch_size = clip_input.dim(0);

        // 1. Encode prompts
        let (pooled_prompt_emb, _clip_hidden) = self
            .clip
            .forward(clip_input)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        let prompt_emb = self
            .t5
            .forward(t5_input)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        // 2. Prepare latents (noise)
        // Flux latents are [B, (H/16)*(W/16), 64]
        let latents_h = height / 16;
        let latents_w = width / 16;
        let latents_seq = latents_h * latents_w;

        let mut latents = if let Some(s) = seed {
            let key =
                mlx_rs::random::key(s).map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
            mlx_rs::random::normal::<f32>(
                &[batch_size, latents_seq as i32, 64],
                None,
                None,
                Some(&key),
            )
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?
        } else {
            mlx_rs::random::normal::<f32>(&[batch_size, latents_seq as i32, 64], None, None, None)
                .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?
        };

        // 3. Prepare IDs for RoPE
        // Text IDs are [B, text_seq, 3] - usually zeros or indices for Flux.
        let text_seq = prompt_emb.dim(1);
        let text_ids = mlx_rs::ops::zeros::<f32>(&[batch_size, text_seq, 3])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        // Image IDs: grid indices for positional encoding
        let mut image_ids_vec = Vec::with_capacity(latents_seq as usize);
        for h in 0..latents_h {
            for w in 0..latents_w {
                image_ids_vec.push([0.0f32, h as f32, w as f32]);
            }
        }
        let image_ids_flat: Vec<f32> = image_ids_vec.into_iter().flatten().collect();
        let image_ids_base = Array::from_slice(&image_ids_flat, &[1, latents_seq as i32, 3]);
        let image_ids = Array::repeat_axis::<f32>(image_ids_base, batch_size as i32, 0)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        // 4. Denoising loop
        let scheduler = FlowMatchScheduler::new_flux(num_steps, 1.0, Some(3.0))?;
        let guidance_arr = Array::from_f32(guidance)
            .expand_dims_axes(&[0])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        let guidance_arr = Array::repeat_axis::<f32>(guidance_arr, batch_size as i32, 0)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        // Scheduler timesteps are already in descending order (high noise → low noise)
        let timesteps = scheduler.timesteps.as_slice::<f32>().to_vec();
        for timestep in timesteps.iter() {
            let t = Array::from_f32(*timestep)
                .expand_dims_axes(&[0])
                .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
            let t_repeated = Array::repeat_axis::<f32>(t, batch_size as i32, 0)
                .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

            let model_output = self
                .dit
                .forward(
                    &latents,
                    &t_repeated,
                    &prompt_emb,
                    &pooled_prompt_emb,
                    Some(&guidance_arr),
                    &text_ids,
                    &image_ids,
                )
                .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

            latents = scheduler.step(&model_output, &t_repeated, &latents)?;
        }

        // 5. Decode latents with VAE (expects NHWC [B, H, W, 16])
        let latents = latents
            .reshape(&[batch_size, latents_h as i32, latents_w as i32, 2, 2, 16])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        let latents = latents
            .transpose_axes(&[0, 1, 3, 2, 4, 5])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        let latents = latents
            .reshape(&[
                batch_size,
                (latents_h * 2) as i32,
                (latents_w * 2) as i32,
                16,
            ])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        let image_nhwc = self
            .vae
            .decode(&latents)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        // Return as NCHW for consistency with standard image formats in ML
        let image = image_nhwc
            .transpose_axes(&[0, 3, 1, 2])
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::zeros;

    #[test]
    fn test_flux_pipeline_structural() {
        let clip_config = CLIPConfig {
            num_layers: 1,
            num_heads: 1,
            ..Default::default()
        };
        let t5_config = T5Config {
            num_layers: 1,
            num_heads: 1,
            ..Default::default()
        };
        let flux_config = FluxConfig {
            num_blocks: 1,
            num_single_blocks: 1,
            ..Default::default()
        };
        let vae_config = VAEConfig::default();

        let mut pipeline = FluxPipeline::new(clip_config, t5_config, flux_config, vae_config);

        let batch = 1;
        let clip_input = zeros::<i32>(&[batch, 77]).unwrap();
        let t5_input = zeros::<i32>(&[batch, 256]).unwrap();

        // Use a very small number of steps for the test
        let out = pipeline.generate(
            &clip_input,
            &t5_input,
            64, // 64x64 small image
            64,
            1, // 1 step
            3.5,
            Some(42),
        );

        assert!(out.is_ok(), "Pipeline failed: {:?}", out.err());
        let image = out.unwrap();
        assert_eq!(image.shape(), &[batch, 3, 64, 64]);
    }
}
