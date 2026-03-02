//! Diffusion sampling and scheduling.
//!
//! Provides schedulers for diffusion models (Flow-Matching, DDIM, etc.)
//! optimized for Apple Silicon.

use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, argmin_axis},
};
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

        let sigmas =
            mlx_rs::ops::linspace::<f32, i32>(sigma_start, sigma_min, num_inference_steps as i32)
                .map_err(|e| PMetalError::Mlx(e.to_string()))?;

        let numerator = sigmas
            .multiply(&Array::from_f32(shift))
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let denominator = Array::from_f32(1.0)
            .add(
                &sigmas
                    .multiply(&Array::from_f32(shift - 1.0))
                    .map_err(|e| PMetalError::Mlx(e.to_string()))?,
            )
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let sigmas = numerator
            .divide(&denominator)
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;

        let timesteps = sigmas
            .multiply(&Array::from_f32(num_train_timesteps))
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;

        Ok(Self {
            sigmas,
            timesteps,
            num_train_timesteps,
        })
    }

    /// Perform a single denoising step.
    pub fn step(&self, model_output: &Array, timestep: &Array, sample: &Array) -> Result<Array> {
        let diff = self
            .timesteps
            .subtract(timestep)
            .map_err(|e| PMetalError::Mlx(e.to_string()))?
            .abs()
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let timestep_id_arr =
            argmin_axis(&diff, 0, false).map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let timestep_id = timestep_id_arr.item::<i32>();

        let sigma = self.sigmas.index(timestep_id).item::<f32>();

        let sigma_next = if (timestep_id as usize) + 1 >= self.sigmas.dim(0) as usize {
            0.0
        } else {
            self.sigmas.index(timestep_id + 1).item::<f32>()
        };

        let delta_sigma = Array::from_f32(sigma_next - sigma);
        let prev_sample = sample
            .add(
                &model_output
                    .multiply(&delta_sigma)
                    .map_err(|e| PMetalError::Mlx(e.to_string()))?,
            )
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;

        Ok(prev_sample)
    }

    /// Add noise to a sample.
    pub fn add_noise(&self, original: &Array, noise: &Array, timestep: &Array) -> Result<Array> {
        let diff = self
            .timesteps
            .subtract(timestep)
            .map_err(|e| PMetalError::Mlx(e.to_string()))?
            .abs()
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let timestep_id_arr =
            argmin_axis(&diff, 0, false).map_err(|e| PMetalError::Mlx(e.to_string()))?;
        let timestep_id = timestep_id_arr.item::<i32>();

        let sigma = self.sigmas.index(timestep_id).item::<f32>();
        let sigma_arr = Array::from_f32(sigma);
        let one_minus_sigma = Array::from_f32(1.0 - sigma);

        let sample = original
            .multiply(&one_minus_sigma)
            .map_err(|e| PMetalError::Mlx(e.to_string()))?
            .add(
                &noise
                    .multiply(&sigma_arr)
                    .map_err(|e| PMetalError::Mlx(e.to_string()))?,
            )
            .map_err(|e| PMetalError::Mlx(e.to_string()))?;
        Ok(sample)
    }
}
