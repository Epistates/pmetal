//! DiLoCo: training that talks to the network once every H steps.
//!
//! Ordinary data parallelism all-reduces gradients every step. On a home
//! cluster joined by Ethernet or Thunderbolt that traffic, not compute, sets
//! the pace. DiLoCo (Douillard et al., 2023, "DiLoCo: Distributed
//! Low-Communication Training of Language Models") lets each worker run `H`
//! ordinary optimizer steps alone, then combines the workers once:
//!
//! 1. every worker starts the round from the same global parameters `θ`;
//! 2. it takes `H` inner steps (AdamW, say) on its own data, reaching `θ_i`;
//! 3. the *pseudo-gradient* `Δ_i = θ - θ_i` is averaged over workers;
//! 4. an outer optimizer (SGD with Nesterov momentum; the paper's `lr = 0.7`,
//!    `momentum = 0.9`) steps the global parameters with it:
//!    `θ ← θ - lr · outer(Δ̄)`, and every worker restarts from the new `θ`.
//!
//! Traffic falls by a factor of `H` (the paper uses `H` in the hundreds), and
//! the pseudo-gradient tolerates the same compression as a gradient
//! ([`GradientCompressor`]); the paper matches fully synchronous training's
//! quality at a fraction of the communication.
//!
//! This type owns the global parameters and the outer momentum on the host,
//! as flat `f32`, and knows nothing about the inner optimizer or the model:
//! the caller copies its parameters into a slice when a round ends and loads
//! the result back.

use crate::compression::GradientCompressor;
use crate::error::DistributedError;
use crate::{DistributedBackend, ReduceOp};
use anyhow::Result;

/// Hyper-parameters of the outer optimizer and the round length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiLoCoConfig {
    /// Inner steps between synchronisations (`H`).
    pub sync_every: usize,
    /// Outer learning rate. The paper uses 0.7 with Nesterov momentum.
    pub outer_lr: f32,
    /// Outer momentum. The paper uses 0.9.
    pub outer_momentum: f32,
    /// Use Nesterov momentum (`g + μ v`) rather than heavy-ball (`v`).
    pub nesterov: bool,
}

impl Default for DiLoCoConfig {
    fn default() -> Self {
        Self {
            sync_every: 500,
            outer_lr: 0.7,
            outer_momentum: 0.9,
            nesterov: true,
        }
    }
}

/// What a synchronisation reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyncStats {
    /// Global L2 norm of the averaged pseudo-gradient.
    pub pseudo_gradient_norm: f64,
    /// Synchronisations done so far, this one included.
    pub round: u64,
    /// Bytes this rank sent for the pseudo-gradient.
    pub bytes_sent: usize,
}

/// The outer loop of DiLoCo for one worker.
pub struct DiLoCo {
    config: DiLoCoConfig,
    /// The global parameters at the start of the current round.
    global: Vec<f32>,
    /// Outer momentum buffer.
    velocity: Vec<f32>,
    steps_this_round: usize,
    rounds: u64,
}

impl DiLoCo {
    /// Start from `initial` parameters. Every worker must start from the
    /// same ones; [`synchronize_initial`](Self::synchronize_initial) makes
    /// them so.
    pub fn new(initial: &[f32], config: DiLoCoConfig) -> Result<Self, DistributedError> {
        if config.sync_every == 0 {
            return Err(DistributedError::Config(
                "DiLoCo sync_every must be at least 1".into(),
            ));
        }
        if !(config.outer_lr.is_finite() && config.outer_momentum.is_finite()) {
            return Err(DistributedError::Config(
                "DiLoCo outer_lr and outer_momentum must be finite".into(),
            ));
        }
        Ok(Self {
            config,
            global: initial.to_vec(),
            velocity: vec![0.0; initial.len()],
            steps_this_round: 0,
            rounds: 0,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &DiLoCoConfig {
        &self.config
    }

    /// The global parameters as of the last synchronisation.
    pub fn global_parameters(&self) -> &[f32] {
        &self.global
    }

    /// Synchronisations completed.
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Make every worker's global parameters rank 0's, so the first round
    /// starts from a common point. `params` is overwritten too.
    pub async fn synchronize_initial(
        &mut self,
        backend: &dyn DistributedBackend,
        params: &mut [f32],
    ) -> Result<()> {
        self.check_len(params.len())?;
        let mut bytes: Vec<u8> = self.global.iter().flat_map(|x| x.to_le_bytes()).collect();
        backend.broadcast(&mut bytes, 0).await?;
        for (g, chunk) in self.global.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *g = f32::from_le_bytes(*chunk);
        }
        params.copy_from_slice(&self.global);
        Ok(())
    }

    /// Record one inner step; true when the round is complete and
    /// [`sync`](Self::sync) should be called.
    pub fn step_done(&mut self) -> bool {
        self.steps_this_round += 1;
        self.steps_this_round >= self.config.sync_every
    }

    /// End a round: average the workers' pseudo-gradients, step the global
    /// parameters with the outer optimizer, and load the result into
    /// `local`, from which the next round starts.
    ///
    /// `local` is this worker's parameters after its inner steps. With a
    /// `compressor` the pseudo-gradient is exchanged compressed (see
    /// [`crate::compression`]); its error feedback carries what a round could
    /// not send into the next. Every worker must call this together.
    pub async fn sync(
        &mut self,
        backend: &dyn DistributedBackend,
        local: &mut [f32],
        compressor: Option<&mut GradientCompressor>,
    ) -> Result<SyncStats> {
        self.check_len(local.len())?;

        // Δ = θ - θ_local: what this worker's inner steps changed, negated,
        // so that it points the way a gradient does.
        let mut delta: Vec<f32> = self
            .global
            .iter()
            .zip(&*local)
            .map(|(g, l)| g - l)
            .collect();
        let bytes_sent = match compressor {
            Some(c) => {
                crate::compression::all_reduce_compressed(backend, c, &mut delta, ReduceOp::Mean)
                    .await?
            }
            None => {
                backend.all_reduce_f32(&mut delta, ReduceOp::Mean).await?;
                delta.len() * 4
            }
        };

        let norm = delta
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        let (mu, lr) = (self.config.outer_momentum, self.config.outer_lr);
        for ((g, v), &d) in self.global.iter_mut().zip(&mut self.velocity).zip(&delta) {
            *v = mu * *v + d;
            let update = if self.config.nesterov {
                d + mu * *v
            } else {
                *v
            };
            *g -= lr * update;
        }
        local.copy_from_slice(&self.global);
        self.steps_this_round = 0;
        self.rounds += 1;
        Ok(SyncStats {
            pseudo_gradient_norm: norm,
            round: self.rounds,
            bytes_sent,
        })
    }

    fn check_len(&self, len: usize) -> Result<(), DistributedError> {
        if len != self.global.len() {
            return Err(DistributedError::Config(format!(
                "DiLoCo holds {} parameters, got {len}",
                self.global.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "diloco_tests.rs"]
mod tests;
