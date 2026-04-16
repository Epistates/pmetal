//! Diffusion sampling and scheduling.
//!
//! Provides schedulers for diffusion models (Flow-Matching, DDIM, etc.)
//! optimized for Apple Silicon.

use pmetal_bridge::compat::ops::argmin_axis;
use pmetal_bridge::compat::{Array, Dtype, ops};
use pmetal_core::{PMetalError, Result};

/// Flow-Matching scheduler for models like Flux.1 and Wan.
#[derive(Debug, Clone)]
pub struct FlowMatchScheduler {
    pub sigmas: Array,
    pub timesteps: Array,
    pub num_train_timesteps: f32,
}

impl FlowMatchScheduler {
    /// Create a new Flow-Matching scheduler for Flux.1.
    pub fn new_flux(
        num_inference_steps: usize,
        denoising_strength: f32,
        shift: Option<f32>,
    ) -> Result<Self> {
        let sigma_min = 0.003 / 1.002;
        let sigma_max = 1.0;
        let shift = shift.unwrap_or(3.0);
        let num_train_timesteps = 1000.0;

        let sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength;

        let sigmas = pmetal_bridge::compat::ops::linspace(
            sigma_start,
            sigma_min,
            num_inference_steps as i32,
            Dtype::Float32,
        );

        let numerator = sigmas.multiply(&Array::from_f32(shift));
        let denominator = Array::from_f32(1.0).add(&sigmas.multiply(&Array::from_f32(shift - 1.0)));
        let sigmas = numerator.divide(&denominator);

        let timesteps = sigmas.multiply(&Array::from_f32(num_train_timesteps));

        // Materialise so downstream as_slice/item calls don't segfault on
        // lazy arrays (data_ptr requires evaluated buffers).
        sigmas.eval();
        timesteps.eval();

        Ok(Self {
            sigmas,
            timesteps,
            num_train_timesteps,
        })
    }

    /// Perform a single denoising step.
    pub fn step(&self, model_output: &Array, timestep: &Array, sample: &Array) -> Result<Array> {
        let diff = self.timesteps.subtract(timestep).abs();
        let timestep_id_arr = argmin_axis(&diff, 0);
        let timestep_id = timestep_id_arr.item::<u32>() as i32;

        let sigma_arr = ops::select_axis(&self.sigmas, timestep_id, 0);
        let sigma = sigma_arr.item::<f32>();

        let sigma_next = if (timestep_id as usize) + 1 >= self.sigmas.dim(0) as usize {
            0.0
        } else {
            let sigma_next_arr = ops::select_axis(&self.sigmas, timestep_id + 1, 0);
            sigma_next_arr.item::<f32>()
        };

        let delta_sigma = Array::from_f32(sigma_next - sigma);
        let prev_sample = sample.add(&model_output.multiply(&delta_sigma));

        Ok(prev_sample)
    }

    /// Add noise to a sample.
    pub fn add_noise(&self, original: &Array, noise: &Array, timestep: &Array) -> Result<Array> {
        let diff = self.timesteps.subtract(timestep).abs();
        let timestep_id_arr = argmin_axis(&diff, 0);
        let timestep_id = timestep_id_arr.item::<u32>() as i32;

        let sigma_arr = ops::select_axis(&self.sigmas, timestep_id, 0);
        let sigma = sigma_arr.item::<f32>();
        let sigma_arr = Array::from_f32(sigma);
        let one_minus_sigma = Array::from_f32(1.0 - sigma);

        let sample = original
            .multiply(&one_minus_sigma)
            .add(&noise.multiply(&sigma_arr));
        Ok(sample)
    }
}
