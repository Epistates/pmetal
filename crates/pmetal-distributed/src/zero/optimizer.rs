//! A sharded optimizer step: reduce-scatter, update the owned shard,
//! all-gather.
//!
//! In plain data parallelism every rank holds the whole optimizer state (for
//! Adam, two more copies of the parameters) and performs the same update. In
//! ZeRO (Rajbhandari et al., 2020) rank `r` holds the state for, and updates,
//! only its `1/N` of the parameters:
//!
//! 1. **reduce-scatter** the gradients, so each rank receives the mean
//!    gradient of its shard only (half the traffic of an all-reduce);
//! 2. update the shard with the local optimizer, whose state covers the shard
//!    and nothing else;
//! 3. **all-gather** the updated shards so every rank has all the parameters.
//!
//! Steps 1 and 3 together move exactly what one all-reduce does, so the
//! saving, `N`-fold less optimizer state and update work per rank, is free in
//! bandwidth. The result is bit-identical to a single device applying the same
//! optimizer to the mean gradient.

use crate::{DistributedBackend, ReduceOp, error::DistributedError};
use anyhow::Result;

/// Which parts of training state are sharded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeROStage {
    /// Optimizer state is sharded. The caller's gradient buffer stays whole.
    Stage1,
    /// Stage 1, and gradients are reduce-scattered into a shard that
    /// replaces the full buffer, so no rank keeps the other ranks' share.
    Stage2,
}

/// An optimizer applied to one rank's shard of the flat parameters.
///
/// `params` and `grads` are the same elements of the flat vector; whatever
/// state the optimizer keeps is for those elements only.
pub trait ShardOptimizer: Send {
    /// Update `params` in place from the mean gradient `grads`.
    fn step(&mut self, params: &mut [f32], grads: &[f32]);

    /// Bytes of optimizer state held, for reporting.
    fn state_bytes(&self) -> usize;
}

/// AdamW (decoupled weight decay) over a shard, on the host.
///
/// Elementwise, so the result for each element does not depend on how the
/// vector is cut into shards. `new(len)` takes the shard's length.
#[derive(Debug, Clone)]
pub struct ShardedAdamW {
    /// Learning rate.
    pub lr: f32,
    /// Exponential decay of the first-moment estimate.
    pub beta1: f32,
    /// Exponential decay of the second-moment estimate.
    pub beta2: f32,
    /// Added to the denominator for stability.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    steps: i32,
}

impl ShardedAdamW {
    /// An optimizer for a shard of `shard_len` elements, with the usual
    /// defaults (betas 0.9 / 0.999, eps 1e-8, weight decay 0.01).
    pub fn new(shard_len: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: vec![0.0; shard_len],
            v: vec![0.0; shard_len],
            steps: 0,
        }
    }
}

impl ShardOptimizer for ShardedAdamW {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), self.m.len(), "shard length changed");
        assert_eq!(grads.len(), self.m.len(), "gradient/shard length mismatch");
        self.steps += 1;
        let c1 = 1.0 - self.beta1.powi(self.steps);
        let c2 = 1.0 - self.beta2.powi(self.steps);
        for (((p, &g), m), v) in params
            .iter_mut()
            .zip(grads)
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            let update = (*m / c1) / ((*v / c2).sqrt() + self.eps);
            *p = *p * (1.0 - self.lr * self.weight_decay) - self.lr * update;
        }
    }

    fn state_bytes(&self) -> usize {
        (self.m.len() + self.v.len()) * std::mem::size_of::<f32>()
    }
}

/// What a sharded step reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroStepStats {
    /// Global L2 norm of the mean gradient, before clipping.
    pub grad_norm: f64,
    /// Factor the gradient was scaled by (1 unless clipped).
    pub clip_scale: f32,
}

/// A sharded optimizer step over a ring (or any backend with
/// `reduce_scatter_f32` and `all_gather_f32_in_place`).
pub struct ZeroOptimizer<O: ShardOptimizer> {
    layout: super::ZeroLayout,
    optimizer: O,
    stage: ZeROStage,
    max_grad_norm: Option<f32>,
}

impl<O: ShardOptimizer> ZeroOptimizer<O> {
    /// Wrap `optimizer`, which must hold state for exactly the layout's
    /// shard.
    pub fn new(layout: super::ZeroLayout, optimizer: O, stage: ZeROStage) -> Self {
        Self {
            layout,
            optimizer,
            stage,
            max_grad_norm: None,
        }
    }

    /// Clip the mean gradient to this global norm before the update.
    pub fn with_max_grad_norm(mut self, max_norm: f32) -> Self {
        self.max_grad_norm = Some(max_norm);
        self
    }

    /// The layout in use.
    pub fn layout(&self) -> &super::ZeroLayout {
        &self.layout
    }

    /// The stage in use.
    pub fn stage(&self) -> ZeROStage {
        self.stage
    }

    /// Bytes of optimizer state this rank holds.
    pub fn state_bytes(&self) -> usize {
        self.optimizer.state_bytes()
    }

    /// One optimizer step.
    ///
    /// `params` is the whole flat parameter vector, identical on every rank;
    /// `grads` is this rank's gradient of it (the same length). On return
    /// `params` holds the updated parameters, identical on every rank and
    /// bit-identical to a single device applying the optimizer to the mean
    /// gradient. `grads` is left with unspecified contents (its owned chunk
    /// holds the mean gradient; under [`ZeROStage::Stage2`] the rest has been
    /// released).
    ///
    /// Every rank must call this together with vectors of the same length.
    pub async fn step(
        &mut self,
        backend: &dyn DistributedBackend,
        params: &mut [f32],
        grads: &mut Vec<f32>,
    ) -> Result<ZeroStepStats> {
        let total = self.layout.total_elements();
        if params.len() != total || grads.len() != total {
            return Err(DistributedError::Config(format!(
                "layout has {total} elements, got {} parameters and {} gradients",
                params.len(),
                grads.len()
            ))
            .into());
        }
        let shard = self.layout.shard_range();

        // 1. Each rank receives the mean gradient of its own shard.
        let owned = backend.reduce_scatter_f32(grads, ReduceOp::Mean).await?;
        if owned != shard {
            return Err(DistributedError::Protocol(format!(
                "the backend scattered chunk {owned:?} to this rank, the layout owns {shard:?}"
            ))
            .into());
        }
        let mut mean_grad: Vec<f32> = grads[shard.clone()].to_vec();
        if self.stage == ZeROStage::Stage2 {
            // The shard is all this rank keeps of the gradient.
            grads.clear();
            grads.shrink_to_fit();
        }

        // 2. Optional global-norm clipping. Each rank's sum of squares is
        // exchanged exactly (as f64 bytes) and added in rank order, so every
        // rank derives the same scale.
        let local_sq: f64 = mean_grad.iter().map(|&g| f64::from(g) * f64::from(g)).sum();
        let sums = backend.all_gather_bytes(&local_sq.to_le_bytes()).await?;
        let norm = sums
            .iter()
            .map(|b| {
                <[u8; 8]>::try_from(b.as_slice())
                    .map(f64::from_le_bytes)
                    .map_err(|_| DistributedError::Protocol("malformed norm contribution".into()))
            })
            .sum::<Result<f64, _>>()?
            .sqrt();
        let mut clip_scale = 1.0f32;
        if let Some(max) = self.max_grad_norm
            && norm > f64::from(max)
        {
            clip_scale = (f64::from(max) / (norm + 1e-6)) as f32;
            mean_grad.iter_mut().for_each(|g| *g *= clip_scale);
        }

        // 3. Update the shard, then give every rank every shard.
        self.optimizer.step(&mut params[shard], &mean_grad);
        backend.all_gather_f32_in_place(params).await?;

        Ok(ZeroStepStats {
            grad_norm: norm,
            clip_scale,
        })
    }
}

#[cfg(test)]
#[path = "optimizer_tests.rs"]
mod tests;
