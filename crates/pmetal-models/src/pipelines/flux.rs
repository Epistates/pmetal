//! The FLUX.1 text-to-image pipeline.
//!
//! `FluxPipeline` of `diffusers`, minus the tokenizers: the CLIP text tower's
//! pooled vector and the T5 encoder's states condition a flow-matching
//! transformer that denoises packed latents, which the VAE decodes.
//!
//! The pipeline takes token ids, as a tokenizer produces them: CLIP's padded or
//! truncated to its 77 positions, T5's padded to the length the release was
//! trained with (512 for dev, 256 for schnell).
//!
//! Latents are channels-last throughout (`[batch, height, width, channels]`,
//! MLX's layout) and the sample stays in float32 across steps whatever the
//! transformer's dtype; only the transformer's inputs are cast.

use pmetal_core::{PMetalError, Result};
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt, ops};
use serde_json::Value;
use std::path::{Path, PathBuf};

use crate::architectures::clip::{CLIPConfig, CLIPTextModel};
use crate::architectures::flux::{FluxConfig, FluxDiT};
use crate::architectures::t5::{T5Config, T5EncoderModel};
use crate::architectures::vae::{AutoencoderKL, VAEConfig};
use crate::loader::{
    load_clip_weights, load_flux_weights, load_t5_weights, load_vae_weights, load_weights,
};
use crate::sampling::diffusion::{
    FlowMatchConfig, FlowMatchEulerDiscreteScheduler, calculate_shift, flux_sigmas,
};

fn model_error(error: impl std::fmt::Display) -> PMetalError {
    PMetalError::ModelLoad(error.to_string())
}

fn runtime_error(error: impl std::fmt::Display) -> PMetalError {
    PMetalError::Mlx(error.to_string())
}

/// Latents are packed in 2x2 patches.
const PATCH: usize = 2;

/// Rearrange latents `[batch, height, width, channels]` into the transformer's
/// token sequence `[batch, (height/2) * (width/2), channels * 4]`: each 2x2
/// patch becomes one token whose channels are `(channel, row, column)`, the
/// channel outermost, as `FluxPipeline._pack_latents` does for channels-first
/// latents.
pub fn pack_latents(latents: &Array) -> Result<Array> {
    let (b, h, w, c) = (
        latents.dim(0),
        latents.dim(1),
        latents.dim(2),
        latents.dim(3),
    );
    if h % 2 != 0 || w % 2 != 0 {
        return Err(PMetalError::Config(format!(
            "flux: latents of {h}x{w} do not divide into 2x2 patches"
        )));
    }
    // [b, h/2, 2, w/2, 2, c] -> [b, h/2, w/2, c, 2, 2]
    Ok(latents
        .reshape(&[b, h / 2, 2, w / 2, 2, c])
        .transpose_axes(&[0, 1, 3, 5, 2, 4])
        .reshape(&[b, (h / 2) * (w / 2), c * 4]))
}

/// The inverse of [`pack_latents`]: tokens `[batch, (height/2) * (width/2),
/// channels * 4]` back to `[batch, height, width, channels]`.
pub fn unpack_latents(tokens: &Array, height: i32, width: i32) -> Result<Array> {
    let (b, n, c4) = (tokens.dim(0), tokens.dim(1), tokens.dim(2));
    if height % 2 != 0 || width % 2 != 0 || n != (height / 2) * (width / 2) || c4 % 4 != 0 {
        return Err(PMetalError::Config(format!(
            "flux: {n} tokens of {c4} channels are not {height}x{width} latents"
        )));
    }
    // [b, h/2, w/2, c, 2, 2] -> [b, h/2, 2, w/2, 2, c]
    Ok(tokens
        .reshape(&[b, height / 2, width / 2, c4 / 4, 2, 2])
        .transpose_axes(&[0, 1, 4, 2, 5, 3])
        .reshape(&[b, height, width, c4 / 4]))
}

/// The position ids of the image tokens: `(0, row, column)` of each 2x2
/// patch, `[(height/2) * (width/2), 3]` in row-major order.
pub fn latent_image_ids(height: i32, width: i32) -> Array {
    let (rows, cols) = (height / 2, width / 2);
    let mut ids = Vec::with_capacity((rows * cols * 3) as usize);
    for row in 0..rows {
        for col in 0..cols {
            ids.extend_from_slice(&[0.0f32, row as f32, col as f32]);
        }
    }
    Array::from_slice(&ids, &[rows * cols, 3])
}

/// What the text encoders produce for a prompt.
#[derive(Debug, Clone)]
pub struct PromptEmbeddings {
    /// T5's states, `[batch, t5 tokens, joint_attention_dim]`.
    pub prompt_embeds: Array,
    /// CLIP's pooled vector, `[batch, pooled_projection_dim]`.
    pub pooled_prompt_embeds: Array,
}

/// FLUX.1 pipeline for text-to-image generation.
pub struct FluxPipeline {
    pub clip: CLIPTextModel,
    pub t5: T5EncoderModel,
    pub dit: FluxDiT,
    pub vae: AutoencoderKL,
    pub scheduler: FlowMatchEulerDiscreteScheduler,
}

impl FluxPipeline {
    /// A pipeline of freshly initialised components.
    pub fn new(
        clip_config: CLIPConfig,
        t5_config: T5Config,
        flux_config: FluxConfig,
        vae_config: VAEConfig,
        scheduler_config: FlowMatchConfig,
    ) -> Result<Self> {
        if flux_config.in_channels != 4 * vae_config.latent_channels {
            return Err(PMetalError::Config(format!(
                "flux: the transformer takes {} channels per patch but the VAE's {} latent \
                 channels make patches of {}",
                flux_config.in_channels,
                vae_config.latent_channels,
                4 * vae_config.latent_channels
            )));
        }
        Ok(Self {
            clip: CLIPTextModel::new(clip_config).map_err(model_error)?,
            t5: T5EncoderModel::new(t5_config).map_err(model_error)?,
            dit: FluxDiT::new(flux_config).map_err(model_error)?,
            vae: AutoencoderKL::new(vae_config).map_err(model_error)?,
            scheduler: FlowMatchEulerDiscreteScheduler::new(scheduler_config)?,
        })
    }

    /// Load the pipeline from a `diffusers` model directory (`transformer/`,
    /// `text_encoder/`, `text_encoder_2/`, `vae/`, `scheduler/`).
    ///
    /// Every component loads strictly: a checkpoint tensor no parameter takes,
    /// a parameter no tensor fills and a tensor of the wrong shape are errors.
    pub fn from_pretrained(model_dir: impl AsRef<Path>) -> Result<Self> {
        let paths = FluxComponentPaths::discover(model_dir.as_ref())?;
        let mut pipeline = Self::new(
            read_config(&paths.text_encoder)?,
            read_config(&paths.text_encoder_2)?,
            read_config(&paths.transformer)?,
            read_config(&paths.vae)?,
            read_scheduler_config(&paths.scheduler)?,
        )?;

        let weights = |dir: &Path| load_weights(dir).map_err(model_error);
        load_clip_weights(&mut pipeline.clip, weights(&paths.text_encoder)?)
            .map_err(model_error)?;
        load_t5_weights(&mut pipeline.t5, weights(&paths.text_encoder_2)?).map_err(model_error)?;
        load_flux_weights(&mut pipeline.dit, weights(&paths.transformer)?).map_err(model_error)?;
        load_vae_weights(&mut pipeline.vae, weights(&paths.vae)?).map_err(model_error)?;

        ModuleParametersExt::eval(&pipeline.clip).map_err(runtime_error)?;
        ModuleParametersExt::eval(&pipeline.t5).map_err(runtime_error)?;
        ModuleParametersExt::eval(&pipeline.dit).map_err(runtime_error)?;
        ModuleParametersExt::eval(&pipeline.vae).map_err(runtime_error)?;
        Ok(pipeline)
    }

    /// The factor between pixels and latent positions: `2^(levels - 1)` for
    /// the VAE, times the 2x2 patching.
    pub fn pixels_per_token(&self) -> usize {
        let levels = self.vae.config.block_out_channels.len();
        PATCH * (1usize << (levels - 1))
    }

    /// Encode a tokenized prompt: CLIP's pooled vector and T5's states.
    pub fn encode_prompt(
        &mut self,
        clip_input: &Array,
        t5_input: &Array,
    ) -> Result<PromptEmbeddings> {
        let pooled_prompt_embeds = self
            .clip
            .forward(clip_input)
            .map_err(runtime_error)?
            .pooled_output;
        let prompt_embeds = self.t5.forward(t5_input, None).map_err(runtime_error)?;
        Ok(PromptEmbeddings {
            prompt_embeds,
            pooled_prompt_embeds,
        })
    }

    /// The latent grid of an image of `height x width` pixels, `(rows,
    /// columns)` before patching; both must be multiples of
    /// [`pixels_per_token`](Self::pixels_per_token).
    fn latent_grid(&self, height: usize, width: usize) -> Result<(i32, i32)> {
        let unit = self.pixels_per_token();
        if height == 0 || width == 0 || height % unit != 0 || width % unit != 0 {
            return Err(PMetalError::Config(format!(
                "flux: an image of {width}x{height} pixels is not a positive multiple of {unit}"
            )));
        }
        Ok((
            (PATCH * (height / unit)) as i32,
            (PATCH * (width / unit)) as i32,
        ))
    }

    /// Denoise `noise` (`[batch, rows, columns, channels]`, standard normal)
    /// into latents under `embeddings`, in `num_steps` Euler steps, and return
    /// them packed.
    ///
    /// `guidance` is the distilled guidance scale of a `dev` model; a model
    /// without a guidance embedder ignores it.
    pub fn sample_latents(
        &mut self,
        embeddings: &PromptEmbeddings,
        noise: &Array,
        num_steps: usize,
        guidance: f32,
    ) -> Result<Array> {
        let batch = noise.dim(0);
        let (rows, cols) = (noise.dim(1), noise.dim(2));
        let dtype = self.dit.x_embedder.weight.value.dtype();
        let f32_ = Dtype::Float32.as_i32();

        let mut latents = pack_latents(&noise.as_dtype(f32_))?;
        let image_tokens = latents.dim(1) as usize;

        let scheduler_config = self.scheduler.config().clone();
        let mu = if scheduler_config.use_dynamic_shifting {
            Some(calculate_shift(
                image_tokens,
                scheduler_config.base_image_seq_len,
                scheduler_config.max_image_seq_len,
                scheduler_config.base_shift.unwrap_or(0.5),
                scheduler_config.max_shift.unwrap_or(1.15),
            ))
        } else {
            None
        };
        self.scheduler
            .set_timesteps(num_steps, Some(&flux_sigmas(num_steps)), mu)?;

        let text_ids = ops::zeros(&[embeddings.prompt_embeds.dim(1), 3], Dtype::Float32);
        let image_ids = latent_image_ids(rows, cols);
        let guidance = self
            .dit
            .config
            .guidance_embeds
            .then(|| ops::full(&[batch], guidance, Dtype::Float32));
        let train_steps = scheduler_config.num_train_timesteps as f32;
        let prompt_embeds = embeddings.prompt_embeds.as_dtype(dtype.as_i32());
        let pooled = embeddings.pooled_prompt_embeds.as_dtype(dtype.as_i32());

        for step in 0..self.scheduler.num_steps() {
            // The transformer takes the noise level itself; it scales to its
            // training timesteps.
            let sigma = self.scheduler.timesteps()[step] / train_steps;
            let timestep = ops::full(&[batch], sigma, Dtype::Float32);
            let velocity = self
                .dit
                .forward(
                    &latents.as_dtype(dtype.as_i32()),
                    &timestep,
                    &prompt_embeds,
                    &pooled,
                    guidance.as_ref(),
                    &text_ids,
                    &image_ids,
                )
                .map_err(runtime_error)?;
            latents = self.scheduler.step(&velocity, step, &latents)?;
            // Bound the graph: one step's worth is held at a time.
            latents.eval();
            pmetal_runtime::check_unobserved_error().map_err(runtime_error)?;
        }
        Ok(latents)
    }

    /// Decode packed latents to an image batch `[batch, 3, height, width]` in
    /// `[-1, 1]`.
    pub fn decode_latents(&mut self, packed: &Array, rows: i32, cols: i32) -> Result<Array> {
        let dtype = self.vae.decoder.conv_in.weight.value.dtype();
        let latents = unpack_latents(packed, rows, cols)?.as_dtype(dtype.as_i32());
        let image = self.vae.decode(&latents).map_err(runtime_error)?;
        Ok(image.transpose_axes(&[0, 3, 1, 2]))
    }

    /// Generate an image batch from tokenized prompts.
    ///
    /// * `clip_input` - `[batch, <= 77]` CLIP token ids.
    /// * `t5_input` - `[batch, tokens]` T5 token ids.
    /// * `width`, `height` - pixels; multiples of
    ///   [`pixels_per_token`](Self::pixels_per_token) (16 for FLUX.1).
    /// * `num_steps` - Euler steps (about 28 for dev, 4 for schnell).
    /// * `guidance` - the distilled guidance scale (about 3.5 for dev).
    /// * `seed` - seed of the initial noise. The same seed and inputs give
    ///   the same image whatever else draws random numbers on the thread,
    ///   and generating leaves the thread's random key alone.
    ///
    /// Returns `[batch, 3, height, width]` in `[-1, 1]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "public API: one argument per setting"
    )]
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
        let (rows, cols) = self.latent_grid(height, width)?;
        let embeddings = self.encode_prompt(clip_input, t5_input)?;
        let channels = (self.dit.config.in_channels / 4) as i32;
        let noise = initial_latents(&[clip_input.dim(0), rows, cols, channels], seed);
        let packed = self.sample_latents(&embeddings, &noise, num_steps, guidance)?;
        self.decode_latents(&packed, rows, cols)
    }
}

/// The noise a generation starts from, standard normal latents of `shape`.
///
/// With a `seed` they are drawn with the key
/// [`draw_key`](pmetal_runtime::decode::draw_key)`(seed, 0)`, as a seeded
/// sampler's first draw is, so a seed gives the same image whatever else
/// draws random numbers on the thread, and the thread's own key is left
/// alone. Without one they take the thread's key.
fn initial_latents(shape: &[i32], seed: Option<u64>) -> Array {
    let key = seed.map(|seed| pmetal_runtime::decode::draw_key(seed, 0));
    pmetal_runtime::compat::random::normal_keyed(shape, Dtype::Float32, key.as_ref())
}

#[derive(Debug, Clone)]
struct FluxComponentPaths {
    transformer: PathBuf,
    text_encoder: PathBuf,
    text_encoder_2: PathBuf,
    vae: PathBuf,
    scheduler: PathBuf,
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
        let component = |name: &str| required_component_dir(model_dir, model_index.as_ref(), name);
        Ok(Self {
            transformer: component("transformer")?,
            text_encoder: component("text_encoder")?,
            text_encoder_2: component("text_encoder_2")?,
            vae: component("vae")?,
            scheduler: model_dir.join("scheduler"),
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
        return Err(PMetalError::Config(format!(
            "Flux model_index.json is missing required component `{component}`"
        )));
    }

    let path = model_dir.join(component);
    if !path.is_dir() {
        return Err(PMetalError::Config(format!(
            "Flux component `{component}` is missing directory {}",
            path.display()
        )));
    }
    if !path.join("config.json").exists() {
        return Err(PMetalError::Config(format!(
            "Flux component `{component}` is missing config.json at {}",
            path.display()
        )));
    }

    Ok(path)
}

fn read_json(path: &Path) -> Result<Value> {
    let raw = std::fs::read_to_string(path).map_err(PMetalError::Io)?;
    serde_json::from_str(&raw).map_err(|e| PMetalError::Serialization(e.to_string()))
}

/// A component's `config.json`, as the config type that names its fields.
fn read_config<T: serde::de::DeserializeOwned>(component_dir: &Path) -> Result<T> {
    let path = component_dir.join("config.json");
    serde_json::from_value(read_json(&path)?)
        .map_err(|e| PMetalError::Config(format!("{}: {e}", path.display())))
}

/// `scheduler/scheduler_config.json`.
fn read_scheduler_config(scheduler_dir: &Path) -> Result<FlowMatchConfig> {
    let path = scheduler_dir.join("scheduler_config.json");
    if !path.exists() {
        return Err(PMetalError::Config(format!(
            "Flux component `scheduler` is missing scheduler_config.json at {}",
            path.display()
        )));
    }
    serde_json::from_value(read_json(&path)?)
        .map_err(|e| PMetalError::Config(format!("{}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tiny_pipeline(guidance_embeds: bool) -> FluxPipeline {
        pmetal_runtime::compat::random::seed(7);
        let clip = CLIPConfig {
            vocab_size: 100,
            embed_dim: 32,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 64,
            max_position_embeddings: 16,
            ..Default::default()
        };
        let t5 = T5Config {
            vocab_size: 100,
            d_model: 32,
            d_ff: 64,
            d_kv: 8,
            num_layers: 1,
            num_heads: 4,
            feed_forward_proj: "gated-gelu".into(),
            ..Default::default()
        };
        let flux = FluxConfig {
            in_channels: 16,
            num_layers: 1,
            num_single_layers: 1,
            attention_head_dim: 16,
            num_attention_heads: 2,
            joint_attention_dim: 32,
            pooled_projection_dim: 32,
            guidance_embeds,
            axes_dims_rope: vec![4, 6, 6],
            ..Default::default()
        };
        let vae = VAEConfig {
            latent_channels: 4,
            block_out_channels: vec![8, 8],
            layers_per_block: 1,
            norm_num_groups: 4,
            ..Default::default()
        };
        FluxPipeline::new(clip, t5, flux, vae, FlowMatchConfig::flux_dev()).unwrap()
    }

    #[test]
    fn test_flux_pipeline_structural() {
        let mut pipeline = tiny_pipeline(true);
        let clip_input = Array::from_slice(&[1i32, 5, 7, 99, 99, 99], &[1, 6]);
        let t5_input = Array::from_slice(&[3i32, 4, 5, 6, 7, 8, 9, 1], &[1, 8]);

        // The VAE has two levels and the patching is 2x2: 4 pixels a token.
        assert_eq!(pipeline.pixels_per_token(), 4);
        let image = pipeline
            .generate(&clip_input, &t5_input, 16, 24, 2, 3.5, Some(42))
            .expect("generates");
        image.eval();
        // A throwing runtime op hands back a placeholder of plausible shape, so
        // the shape alone can't show the pipeline ran.
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
        assert_eq!(image.shape(), &[1, 3, 24, 16]);
        let again = pipeline
            .generate(&clip_input, &t5_input, 16, 24, 2, 3.5, Some(42))
            .unwrap();
        let values = |a: &Array| pmetal_cache::test_utils::to_f32_vec_eval(a);
        assert_eq!(values(&image), values(&again), "same seed, same image");
        let other = pipeline
            .generate(&clip_input, &t5_input, 16, 24, 2, 3.5, Some(43))
            .unwrap();
        assert_ne!(
            values(&image),
            values(&other),
            "another seed, another image"
        );
        assert!(values(&image).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn an_image_that_is_not_a_multiple_of_the_patch_is_refused() {
        let mut pipeline = tiny_pipeline(false);
        let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
        let err = pipeline
            .generate(&ids, &ids, 18, 16, 1, 0.0, Some(1))
            .expect_err("18 is not a multiple of 4");
        assert!(err.to_string().contains("18"), "{err}");
    }

    /// A seeded generation's noise is a function of its seed alone: other
    /// draws on the thread between two generations don't change it, and
    /// drawing it neither reseeds nor advances the thread's key.
    #[test]
    #[serial_test::serial]
    fn seeded_initial_noise_is_independent_of_the_thread_key() {
        use pmetal_runtime::compat::{Dtype, random};

        let shape = [1, 4, 4, 4];
        let floats = |a: Array| {
            a.eval();
            a.as_slice::<f32>().to_vec()
        };
        let thread_draw = || floats(random::uniform(&[8], Dtype::Float32));

        random::seed(1);
        let a = floats(initial_latents(&shape, Some(42)));
        let after_a = thread_draw();
        random::seed(99);
        let _ = thread_draw();
        assert_eq!(
            a,
            floats(initial_latents(&shape, Some(42))),
            "the same seed drew different noise"
        );

        // The thread's key went on from where it was, untouched.
        random::seed(1);
        assert_eq!(
            thread_draw(),
            after_a,
            "the seeded draw moved the thread's key"
        );

        assert_ne!(
            a,
            floats(initial_latents(&shape, Some(43))),
            "another seed drew the same noise"
        );
        // Unseeded, the noise is the thread's.
        random::seed(5);
        let unseeded = floats(initial_latents(&shape, None));
        random::seed(5);
        assert_eq!(unseeded, floats(initial_latents(&shape, None)));
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    }

    #[test]
    fn a_transformer_that_does_not_fit_the_vae_is_refused() {
        let err = FluxPipeline::new(
            CLIPConfig::default(),
            T5Config::default(),
            FluxConfig {
                in_channels: 64,
                ..Default::default()
            },
            VAEConfig {
                latent_channels: 4,
                ..Default::default()
            },
            FlowMatchConfig::default(),
        )
        .err()
        .expect("64 channels do not make 4-channel patches");
        assert!(err.to_string().contains("latent"), "{err}");
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

    #[test]
    fn a_missing_scheduler_config_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_scheduler_config(&dir.path().join("scheduler")).expect_err("no config");
        assert!(err.to_string().contains("scheduler_config.json"), "{err}");
    }
}
