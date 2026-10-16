//! Diffusion sampling and scheduling.
//!
//! The flow-matching Euler scheduler of rectified-flow models (FLUX.1, SD3),
//! after `diffusers`' `FlowMatchEulerDiscreteScheduler`: the model predicts the
//! velocity `v = noise - data` of the straight path between data and noise, a
//! step moves the sample along it by the gap between consecutive noise levels
//! (sigmas), and the schedule of sigmas is the one thing a model release tunes
//! (a static `shift`, or a shift that grows with the image's token count).
//!
//! The schedule is host data, a few dozen floats; only the sample and the
//! model output are arrays. The scheduler is stateless between steps: `step`
//! takes the step's index.

use pmetal_core::{PMetalError, Result};
use pmetal_runtime::compat::{Array, Dtype};
use serde::{Deserialize, Serialize};

/// How the resolution-dependent shift `mu` bends the schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeShiftType {
    /// `e^mu / (e^mu + (1/t - 1))`, FLUX's.
    #[default]
    Exponential,
    /// `mu / (mu + (1/t - 1))`.
    Linear,
}

/// The scheduler's `scheduler_config.json`, with the defaults of the
/// `diffusers` class. Settings this scheduler does not implement are refused
/// when set rather than ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMatchConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    /// A static shift of the whole schedule, `s t / (1 + (s - 1) t)`. Unused
    /// when `use_dynamic_shifting`.
    #[serde(default = "default_shift")]
    pub shift: f32,
    /// Shift by an amount `mu` the caller computes from the image's token
    /// count ([`calculate_shift`]).
    #[serde(default)]
    pub use_dynamic_shifting: bool,
    #[serde(default = "default_base_shift")]
    pub base_shift: Option<f64>,
    #[serde(default = "default_max_shift")]
    pub max_shift: Option<f64>,
    #[serde(default = "default_base_image_seq_len")]
    pub base_image_seq_len: usize,
    #[serde(default = "default_max_image_seq_len")]
    pub max_image_seq_len: usize,
    /// Stretch the shifted schedule so that it ends at this sigma.
    #[serde(default)]
    pub shift_terminal: Option<f32>,
    #[serde(default)]
    pub time_shift_type: TimeShiftType,
    #[serde(default)]
    pub invert_sigmas: bool,
    #[serde(default)]
    pub use_karras_sigmas: bool,
    #[serde(default)]
    pub use_exponential_sigmas: bool,
    #[serde(default)]
    pub use_beta_sigmas: bool,
    #[serde(default)]
    pub stochastic_sampling: bool,
}

fn default_num_train_timesteps() -> usize {
    1000
}
fn default_shift() -> f32 {
    1.0
}
fn default_base_shift() -> Option<f64> {
    Some(0.5)
}
fn default_max_shift() -> Option<f64> {
    Some(1.15)
}
fn default_base_image_seq_len() -> usize {
    256
}
fn default_max_image_seq_len() -> usize {
    4096
}

impl Default for FlowMatchConfig {
    /// `diffusers`' `FlowMatchEulerDiscreteScheduler()`: no shift.
    fn default() -> Self {
        Self {
            num_train_timesteps: default_num_train_timesteps(),
            shift: default_shift(),
            use_dynamic_shifting: false,
            base_shift: default_base_shift(),
            max_shift: default_max_shift(),
            base_image_seq_len: default_base_image_seq_len(),
            max_image_seq_len: default_max_image_seq_len(),
            shift_terminal: None,
            time_shift_type: TimeShiftType::default(),
            invert_sigmas: false,
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
            stochastic_sampling: false,
        }
    }
}

impl FlowMatchConfig {
    /// FLUX.1-schnell: the schedule is used as given.
    pub fn flux_schnell() -> Self {
        Self::default()
    }

    /// FLUX.1-dev: a shift that grows with the number of image tokens.
    pub fn flux_dev() -> Self {
        Self {
            shift: 3.0,
            use_dynamic_shifting: true,
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<()> {
        let unsupported = [
            (self.invert_sigmas, "invert_sigmas"),
            (self.use_karras_sigmas, "use_karras_sigmas"),
            (self.use_exponential_sigmas, "use_exponential_sigmas"),
            (self.use_beta_sigmas, "use_beta_sigmas"),
            (self.stochastic_sampling, "stochastic_sampling"),
        ];
        if let Some((_, name)) = unsupported.iter().find(|(set, _)| *set) {
            return Err(PMetalError::Config(format!(
                "flow-match scheduler: `{name}` is not supported"
            )));
        }
        if self.num_train_timesteps == 0 {
            return Err(PMetalError::Config(
                "flow-match scheduler: num_train_timesteps is zero".into(),
            ));
        }
        Ok(())
    }
}

/// The shift `mu` for an image of `image_seq_len` tokens: linear in the token
/// count between `base_shift` at `base_seq_len` and `max_shift` at
/// `max_seq_len`, as FLUX's pipeline computes it.
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let slope = (max_shift - base_shift) / (max_seq_len as f64 - base_seq_len as f64);
    let intercept = base_shift - slope * base_seq_len as f64;
    image_seq_len as f64 * slope + intercept
}

/// The noise levels `1, (n-1)/n, ..., 1/n` that FLUX's pipeline hands the
/// scheduler for `n` steps (before shifting).
pub fn flux_sigmas(num_inference_steps: usize) -> Vec<f32> {
    let n = num_inference_steps;
    linspace(1.0, 1.0 / n as f64, n)
        .into_iter()
        .map(|sigma| sigma as f32)
        .collect()
}

/// `numpy.linspace(start, stop, n)`: `start + i * step`, with the last point
/// exactly `stop`.
fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (n - 1) as f64;
            let mut points: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
            points[n - 1] = stop;
            points
        }
    }
}

/// Flow-matching Euler scheduler.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    config: FlowMatchConfig,
    /// The sigmas of the current schedule, followed by the terminal `0`.
    sigmas: Vec<f32>,
    /// `sigma * num_train_timesteps` for each step (one fewer than `sigmas`).
    timesteps: Vec<f32>,
    /// The noise range of the training schedule, for the default sigmas.
    sigma_min: f32,
    sigma_max: f32,
}

impl FlowMatchEulerDiscreteScheduler {
    pub fn new(config: FlowMatchConfig) -> Result<Self> {
        config.validate()?;
        // The training schedule `1, (T-1)/T, ..., 1/T`, shifted unless the
        // shift is dynamic; its ends bound the default inference schedule.
        let t = config.num_train_timesteps as f32;
        let shifted = |sigma: f32| {
            if config.use_dynamic_shifting {
                sigma
            } else {
                config.shift * sigma / (1.0 + (config.shift - 1.0) * sigma)
            }
        };
        let (sigma_max, sigma_min) = (shifted(1.0), shifted(1.0 / t));
        let mut scheduler = Self {
            config,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            sigma_min,
            sigma_max,
        };
        scheduler.sigmas = vec![sigma_max, 0.0];
        scheduler.timesteps = vec![sigma_max * t];
        Ok(scheduler)
    }

    pub fn config(&self) -> &FlowMatchConfig {
        &self.config
    }

    /// The sigmas of the current schedule, with the terminal `0` appended.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// The model timesteps of the current schedule, `sigma * 1000` for the
    /// usual training length.
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    /// The number of steps in the current schedule.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    fn time_shift(&self, mu: f64, t: f32) -> f32 {
        let ratio = 1.0 / t - 1.0;
        match self.config.time_shift_type {
            TimeShiftType::Exponential => {
                let e = mu.exp() as f32;
                e / (e + ratio)
            }
            TimeShiftType::Linear => {
                let mu = mu as f32;
                mu / (mu + ratio)
            }
        }
    }

    /// Lay out `num_inference_steps` steps.
    ///
    /// `sigmas` are the noise levels before shifting (FLUX passes
    /// [`flux_sigmas`]); without them the schedule runs evenly between the
    /// training schedule's ends. `mu` is required when the config shifts
    /// dynamically (see [`calculate_shift`]).
    pub fn set_timesteps(
        &mut self,
        num_inference_steps: usize,
        sigmas: Option<&[f32]>,
        mu: Option<f64>,
    ) -> Result<()> {
        if self.config.use_dynamic_shifting && mu.is_none() {
            return Err(PMetalError::Config(
                "flow-match scheduler: `mu` is required when `use_dynamic_shifting` is set".into(),
            ));
        }
        let t = self.config.num_train_timesteps as f32;
        let mut sigmas: Vec<f32> = match sigmas {
            Some(sigmas) => {
                if sigmas.len() != num_inference_steps {
                    return Err(PMetalError::Config(format!(
                        "flow-match scheduler: {} sigmas for {num_inference_steps} steps",
                        sigmas.len()
                    )));
                }
                sigmas.to_vec()
            }
            None => {
                // numpy.linspace(sigma_max * T, sigma_min * T, n) / T, in f64.
                linspace(
                    f64::from(self.sigma_max) * f64::from(t),
                    f64::from(self.sigma_min) * f64::from(t),
                    num_inference_steps,
                )
                .into_iter()
                .map(|timestep| (timestep / f64::from(t)) as f32)
                .collect()
            }
        };

        if self.config.use_dynamic_shifting {
            let mu = mu.unwrap_or_default();
            for sigma in &mut sigmas {
                *sigma = self.time_shift(mu, *sigma);
            }
        } else {
            let shift = self.config.shift;
            for sigma in &mut sigmas {
                *sigma = shift * *sigma / (1.0 + (shift - 1.0) * *sigma);
            }
        }

        if let Some(terminal) = self.config.shift_terminal.filter(|s| *s != 0.0) {
            // Stretch `1 - sigma` so that the last sigma lands on `terminal`.
            let last = 1.0 - sigmas.last().copied().unwrap_or(0.0);
            let scale = last / (1.0 - terminal);
            for sigma in &mut sigmas {
                *sigma = 1.0 - (1.0 - *sigma) / scale;
            }
        }

        self.timesteps = sigmas.iter().map(|s| s * t).collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        Ok(())
    }

    /// One Euler step: `sample + (sigma_next - sigma) * model_output`, computed
    /// in f32 and returned in f32 whatever the dtypes of the inputs, so that
    /// the accumulation over steps does not round to a half-precision model's
    /// dtype each time; the caller casts to the model's dtype for its next
    /// call.
    pub fn step(&self, model_output: &Array, step_index: usize, sample: &Array) -> Result<Array> {
        let (sigma, next) = self.sigma_pair(step_index)?;
        let f32_ = Dtype::Float32.as_i32();
        Ok(sample
            .as_dtype(f32_)
            .add(&model_output.as_dtype(f32_).mul_scalar(next - sigma)))
    }

    /// `sigma * noise + (1 - sigma) * sample`: `sample` noised to the level
    /// of step `step_index` (image-to-image's starting point).
    pub fn scale_noise(&self, sample: &Array, noise: &Array, step_index: usize) -> Result<Array> {
        let (sigma, _) = self.sigma_pair(step_index)?;
        let f32_ = Dtype::Float32.as_i32();
        Ok(noise
            .as_dtype(f32_)
            .mul_scalar(sigma)
            .add(&sample.as_dtype(f32_).mul_scalar(1.0 - sigma)))
    }

    fn sigma_pair(&self, step_index: usize) -> Result<(f32, f32)> {
        match (self.sigmas.get(step_index), self.sigmas.get(step_index + 1)) {
            (Some(&sigma), Some(&next)) => Ok((sigma, next)),
            _ => Err(PMetalError::Config(format!(
                "flow-match scheduler: step {step_index} of {}",
                self.timesteps.len()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_sigmas_are_numpy_linspace() {
        assert_eq!(flux_sigmas(1), vec![1.0]);
        let four = flux_sigmas(4);
        for (got, want) in four.iter().zip([1.0, 0.75, 0.5, 0.25]) {
            assert!((got - want).abs() < 1e-7, "{four:?}");
        }
    }

    #[test]
    fn mu_runs_from_base_to_max_shift() {
        assert!((calculate_shift(256, 256, 4096, 0.5, 1.15) - 0.5).abs() < 1e-12);
        assert!((calculate_shift(4096, 256, 4096, 0.5, 1.15) - 1.15).abs() < 1e-6);
    }

    #[test]
    fn dynamic_shifting_needs_mu_and_unsupported_modes_are_refused() {
        let mut dev = FlowMatchEulerDiscreteScheduler::new(FlowMatchConfig::flux_dev()).unwrap();
        assert!(dev.set_timesteps(4, Some(&flux_sigmas(4)), None).is_err());
        assert!(
            FlowMatchEulerDiscreteScheduler::new(FlowMatchConfig {
                use_karras_sigmas: true,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn a_step_index_past_the_schedule_is_an_error() {
        let mut s = FlowMatchEulerDiscreteScheduler::new(FlowMatchConfig::default()).unwrap();
        s.set_timesteps(3, Some(&flux_sigmas(3)), None).unwrap();
        let x = Array::from_slice(&[1.0f32], &[1]);
        assert!(s.step(&x, 2, &x).is_ok());
        assert!(s.step(&x, 3, &x).is_err());
    }
}
