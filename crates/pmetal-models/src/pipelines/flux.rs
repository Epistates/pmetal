//! Flux.1 inference pipeline.
//!
//! Coordinates CLIP/T5 text encoders, Flux DiT, and VAE for end-to-end
//! high-quality image generation on Apple Silicon.

use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, ModuleParametersExt},
};
use pmetal_core::Result;
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::architectures::clip::{CLIPConfig, CLIPTextModel};
use crate::architectures::flux::{FluxConfig, FluxDiT};
use crate::architectures::t5::{T5Config, T5EncoderModel};
use crate::architectures::vae::{FluxVAE, VAEConfig};
use crate::loader::{
    load_clip_weights, load_flux_weights, load_t5_weights, load_vae_weights, load_weights,
};
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
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let component_paths = FluxComponentPaths::discover(model_dir)?;

        let clip_config = load_clip_config(&component_paths.text_encoder)?;
        let t5_config = load_t5_config(&component_paths.text_encoder_2)?;
        let flux_config = load_flux_config(&component_paths.transformer)?;
        let vae_config = load_vae_config(&component_paths.vae)?;

        let mut pipeline = Self::new(clip_config, t5_config, flux_config, vae_config);

        let clip_weights = load_weights(&component_paths.text_encoder)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;
        load_clip_weights(&mut pipeline.clip, &clip_weights)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;

        let t5_weights = load_weights(&component_paths.text_encoder_2)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;
        load_t5_weights(&mut pipeline.t5, &t5_weights)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;

        let flux_weights = load_weights(&component_paths.transformer)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;
        load_flux_weights(&mut pipeline.dit, &flux_weights)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;

        let vae_weights = load_weights(&component_paths.vae)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;
        load_vae_weights(&mut pipeline.vae, &vae_weights)
            .map_err(|e| pmetal_core::PMetalError::ModelLoad(e.to_string()))?;

        ModuleParametersExt::eval(&pipeline.clip)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        ModuleParametersExt::eval(&pipeline.t5)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        ModuleParametersExt::eval(&pipeline.dit)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;
        ModuleParametersExt::eval(&pipeline.vae)
            .map_err(|e| pmetal_core::PMetalError::Mlx(e.to_string()))?;

        Ok(pipeline)
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

#[derive(Debug, Clone)]
struct FluxComponentPaths {
    transformer: PathBuf,
    text_encoder: PathBuf,
    text_encoder_2: PathBuf,
    vae: PathBuf,
}

impl FluxComponentPaths {
    fn discover(model_dir: &Path) -> Result<Self> {
        let model_index = {
            let path = model_dir.join("model_index.json");
            if path.exists() {
                Some(read_json(&path)?)
            } else {
                None
            }
        };

        let transformer = required_component_dir(model_dir, model_index.as_ref(), "transformer")?;
        let text_encoder = required_component_dir(model_dir, model_index.as_ref(), "text_encoder")?;
        let text_encoder_2 =
            required_component_dir(model_dir, model_index.as_ref(), "text_encoder_2")?;
        let vae = required_component_dir(model_dir, model_index.as_ref(), "vae")?;

        Ok(Self {
            transformer,
            text_encoder,
            text_encoder_2,
            vae,
        })
    }
}

fn required_component_dir(
    model_dir: &Path,
    model_index: Option<&Value>,
    component: &str,
) -> Result<PathBuf> {
    if let Some(model_index) = model_index
        && model_index.get(component).is_none()
    {
        return Err(pmetal_core::PMetalError::Config(format!(
            "Flux model_index.json is missing required component `{component}`"
        )));
    }

    let path = model_dir.join(component);
    if !path.is_dir() {
        return Err(pmetal_core::PMetalError::Config(format!(
            "Flux component `{component}` is missing directory {}",
            path.display()
        )));
    }
    if !path.join("config.json").exists() {
        return Err(pmetal_core::PMetalError::Config(format!(
            "Flux component `{component}` is missing config.json at {}",
            path.display()
        )));
    }

    Ok(path)
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = std::fs::read_to_string(path).map_err(pmetal_core::PMetalError::Io)?;
    serde_json::from_str(&raw).map_err(|e| pmetal_core::PMetalError::Serialization(e.to_string()))
}

fn load_clip_config(component_dir: &Path) -> Result<CLIPConfig> {
    let raw = read_json(&component_dir.join("config.json"))?;
    let mut config = CLIPConfig::default();

    config.vocab_size = value_usize(&raw, &["vocab_size"]).unwrap_or(config.vocab_size);
    config.embed_dim = value_usize(&raw, &["embed_dim", "hidden_size"]).unwrap_or(config.embed_dim);
    config.num_layers =
        value_usize(&raw, &["num_layers", "num_hidden_layers"]).unwrap_or(config.num_layers);
    config.num_heads =
        value_usize(&raw, &["num_heads", "num_attention_heads"]).unwrap_or(config.num_heads);
    config.intermediate_size =
        value_usize(&raw, &["intermediate_size"]).unwrap_or(config.intermediate_size);
    config.max_position_embeddings =
        value_usize(&raw, &["max_position_embeddings"]).unwrap_or(config.max_position_embeddings);
    config.layer_norm_eps = value_f32(&raw, &["layer_norm_eps"]).unwrap_or(config.layer_norm_eps);
    config.use_quick_gelu = match value_str(&raw, &["hidden_act", "activation_function"]) {
        Some("quick_gelu") => true,
        Some("gelu") | Some("gelu_new") => false,
        _ => config.use_quick_gelu,
    };

    Ok(config)
}

fn load_t5_config(component_dir: &Path) -> Result<T5Config> {
    let raw = read_json(&component_dir.join("config.json"))?;
    let mut config = T5Config::default();

    config.vocab_size = value_usize(&raw, &["vocab_size"]).unwrap_or(config.vocab_size);
    config.d_model = value_usize(&raw, &["d_model"]).unwrap_or(config.d_model);
    config.d_ff = value_usize(&raw, &["d_ff"]).unwrap_or(config.d_ff);
    config.d_kv = value_usize(&raw, &["d_kv"]).unwrap_or(config.d_kv);
    config.num_layers =
        value_usize(&raw, &["num_layers", "num_hidden_layers"]).unwrap_or(config.num_layers);
    config.num_heads =
        value_usize(&raw, &["num_heads", "num_attention_heads"]).unwrap_or(config.num_heads);
    config.relative_attention_num_buckets = value_usize(&raw, &["relative_attention_num_buckets"])
        .unwrap_or(config.relative_attention_num_buckets);
    config.relative_attention_max_distance =
        value_usize(&raw, &["relative_attention_max_distance"])
            .unwrap_or(config.relative_attention_max_distance);
    config.dropout_rate = value_f32(&raw, &["dropout_rate"]).unwrap_or(config.dropout_rate);
    config.layer_norm_epsilon =
        value_f32(&raw, &["layer_norm_epsilon"]).unwrap_or(config.layer_norm_epsilon);
    if let Some(feed_forward_proj) = value_str(&raw, &["feed_forward_proj"]) {
        config.feed_forward_proj = feed_forward_proj.to_string();
        config.is_gated_act = feed_forward_proj.contains("gated");
    }

    Ok(config)
}

fn load_vae_config(component_dir: &Path) -> Result<VAEConfig> {
    let raw = read_json(&component_dir.join("config.json"))?;
    let mut config = VAEConfig::default();

    config.in_channels = value_usize(&raw, &["in_channels"]).unwrap_or(config.in_channels);
    config.out_channels = value_usize(&raw, &["out_channels"]).unwrap_or(config.out_channels);
    config.latent_channels =
        value_usize(&raw, &["latent_channels"]).unwrap_or(config.latent_channels);
    config.layers_per_block =
        value_usize(&raw, &["layers_per_block"]).unwrap_or(config.layers_per_block);
    if let Some(block_out_channels) = value_usize_array(&raw, &["block_out_channels"]) {
        config.block_out_channels = block_out_channels;
    }

    Ok(config)
}

fn load_flux_config(component_dir: &Path) -> Result<FluxConfig> {
    let raw = read_json(&component_dir.join("config.json"))?;
    let mut config = FluxConfig::default();

    config.input_dim = value_usize(&raw, &["input_dim", "in_channels"]).unwrap_or(config.input_dim);
    config.num_attention_heads =
        value_usize(&raw, &["num_attention_heads"]).unwrap_or(config.num_attention_heads);
    let attention_head_dim = value_usize(&raw, &["attention_head_dim"]);
    config.hidden_size = value_usize(&raw, &["hidden_size"])
        .or_else(|| attention_head_dim.map(|dim| dim * config.num_attention_heads))
        .unwrap_or(config.hidden_size);
    config.num_blocks =
        value_usize(&raw, &["num_blocks", "num_layers"]).unwrap_or(config.num_blocks);
    config.num_single_blocks = value_usize(&raw, &["num_single_blocks", "num_single_layers"])
        .unwrap_or(config.num_single_blocks);
    config.rope_theta = value_f32(&raw, &["rope_theta", "theta"]).unwrap_or(config.rope_theta);
    if let Some(axes_dim) = value_usize_array(&raw, &["axes_dim", "axes_dims_rope"]) {
        config.axes_dim = axes_dim;
    }
    config.disable_guidance_embedder = value_bool(&raw, &["disable_guidance_embedder"])
        .unwrap_or_else(|| {
            value_bool(&raw, &["guidance_embeds"])
                .map(|enabled| !enabled)
                .unwrap_or(config.disable_guidance_embedder)
        });
    config.timestep_dim = value_usize(&raw, &["timestep_dim"]).unwrap_or(config.timestep_dim);
    config.pooled_embed_dim = value_usize(&raw, &["pooled_embed_dim", "pooled_projection_dim"])
        .unwrap_or(config.pooled_embed_dim);
    config.context_embed_dim = value_usize(&raw, &["context_embed_dim", "joint_attention_dim"])
        .unwrap_or(config.context_embed_dim);
    config.norm_epsilon = value_f32(&raw, &["norm_epsilon"]).unwrap_or(config.norm_epsilon);

    Ok(config)
}

fn value_usize(raw: &Value, keys: &[&str]) -> Option<usize> {
    keys.iter()
        .find_map(|key| raw.get(*key).and_then(Value::as_u64))
        .map(|v| v as usize)
}

fn value_f32(raw: &Value, keys: &[&str]) -> Option<f32> {
    keys.iter()
        .find_map(|key| raw.get(*key).and_then(Value::as_f64))
        .map(|v| v as f32)
}

fn value_bool(raw: &Value, keys: &[&str]) -> Option<bool> {
    keys.iter()
        .find_map(|key| raw.get(*key).and_then(Value::as_bool))
}

fn value_str<'a>(raw: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| raw.get(*key).and_then(Value::as_str))
}

fn value_usize_array(raw: &Value, keys: &[&str]) -> Option<Vec<usize>> {
    keys.iter().find_map(|key| {
        raw.get(*key).and_then(Value::as_array).map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .map(|v| v as usize)
                .collect::<Vec<_>>()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::zeros;
    use serde_json::json;

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

    #[test]
    fn test_load_flux_config_from_diffusers_style_json() {
        let raw = json!({
            "in_channels": 64,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "num_layers": 19,
            "num_single_layers": 38,
            "axes_dims_rope": [16, 56, 56],
            "guidance_embeds": true,
            "pooled_projection_dim": 768,
            "joint_attention_dim": 4096,
        });

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&raw).unwrap(),
        )
        .unwrap();

        let config = load_flux_config(dir.path()).unwrap();
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.hidden_size, 24 * 128);
        assert_eq!(config.num_blocks, 19);
        assert_eq!(config.num_single_blocks, 38);
        assert!(!config.disable_guidance_embedder);
        assert_eq!(config.pooled_embed_dim, 768);
        assert_eq!(config.context_embed_dim, 4096);
    }

    #[test]
    fn test_flux_component_discovery_uses_model_index() {
        let dir = tempfile::tempdir().unwrap();
        for component in ["transformer", "text_encoder", "text_encoder_2", "vae"] {
            let component_dir = dir.path().join(component);
            std::fs::create_dir(&component_dir).unwrap();
            std::fs::write(component_dir.join("config.json"), "{}").unwrap();
        }

        let model_index = json!({
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "vae": ["diffusers", "AutoencoderKL"]
        });
        std::fs::write(
            dir.path().join("model_index.json"),
            serde_json::to_string(&model_index).unwrap(),
        )
        .unwrap();

        let paths = FluxComponentPaths::discover(dir.path()).unwrap();
        assert_eq!(paths.transformer, dir.path().join("transformer"));
        assert_eq!(paths.text_encoder, dir.path().join("text_encoder"));
        assert_eq!(paths.text_encoder_2, dir.path().join("text_encoder_2"));
        assert_eq!(paths.vae, dir.path().join("vae"));
    }
}
