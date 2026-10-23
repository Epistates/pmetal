//! AdamW on the host: the scalars of a step and the update of a slice.
//!
//! The device AdamW (`pmetal_runtime::optimizer::AdamW`, one fused Metal
//! kernel per parameter) and this host update are one piece of arithmetic: the
//! same operations in the same order, in f32, as `torch.optim.AdamW` computes
//! them,
//!
//! ```text
//! p = p·decay                           decoupled weight decay, on the old p
//! m = m + (1−β₁)·(g − m)
//! v = β₂·v + (1−β₂)·g²
//! p = p − step_size · m / (√v / bias_correction2_sqrt + ε)
//! ```
//!
//! so the two agree to the last bit. The runtime sits below this crate and
//! cannot call it, which is why it keeps a scalars type of its own; its tests
//! hold the two together.
//!
//! This is what the trainers that keep their weights in host memory run: the
//! ANE trainer and the sharded optimizer of `pmetal-distributed`.

/// The scalars of AdamW step `t`, computed as `torch.optim.AdamW` computes
/// them: in f64, each rounded to f32 once.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWStepScalars {
    /// `1 − lr·λ`: the decoupled weight decay, applied before the update.
    pub decay: f32,
    /// `1 − β₁`, the weight the first moment moves toward the gradient by.
    pub one_minus_beta1: f32,
    /// `β₂`.
    pub beta2: f32,
    /// `1 − β₂`.
    pub one_minus_beta2: f32,
    /// `lr / (1 − β₁ᵗ)`.
    pub step_size: f32,
    /// `√(1 − β₂ᵗ)`.
    pub bias_correction2_sqrt: f32,
    /// `ε`, added after the second moment's bias correction.
    pub eps: f32,
}

impl AdamWStepScalars {
    /// The scalars of 1-based step `t`.
    ///
    /// # Panics
    /// If `t` is 0: Adam's bias correction is undefined before the first step.
    pub fn new(lr: f32, weight_decay: f32, beta1: f32, beta2: f32, eps: f32, t: u64) -> Self {
        assert!(t >= 1, "AdamW steps are 1-based; advance the step first");
        let (lr, wd, b1, b2) = (
            f64::from(lr),
            f64::from(weight_decay),
            f64::from(beta1),
            f64::from(beta2),
        );
        // `powf` rather than `powi`: a step count past i32::MAX must not wrap.
        let t = t as f64;
        Self {
            decay: (1.0 - lr * wd) as f32,
            one_minus_beta1: (1.0 - b1) as f32,
            beta2,
            one_minus_beta2: (1.0 - b2) as f32,
            step_size: (lr / (1.0 - b1.powf(t))) as f32,
            bias_correction2_sqrt: (1.0 - b2.powf(t)).sqrt() as f32,
            eps,
        }
    }
}

/// One AdamW step over a parameter held on the host: `w` by the gradient `g`,
/// with moments `m` and `v`.
///
/// # Panics
/// If the slices differ in length.
pub fn adamw_update(w: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], s: &AdamWStepScalars) {
    let n = w.len();
    assert_eq!(n, g.len(), "gradient length");
    assert_eq!(n, m.len(), "first moment length");
    assert_eq!(n, v.len(), "second moment length");

    for i in 0..n {
        let gi = g[i];
        let mi = m[i] + s.one_minus_beta1 * (gi - m[i]);
        let vi = v[i] * s.beta2 + gi * gi * s.one_minus_beta2;
        m[i] = mi;
        v[i] = vi;
        let denom = vi.sqrt() / s.bias_correction2_sqrt + s.eps;
        w[i] = w[i] * s.decay - mi / denom * s.step_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_step_moves_each_weight_by_the_learning_rate() {
        // m̂/√v̂ = g/|g| on the first step, so each weight moves by lr after
        // its decay: p·(1 − lr·λ) − lr·sign(g).
        let mut w = vec![1.0f32, 2.0, 3.0];
        let g = vec![0.1, -0.2, 0.3];
        let (mut m, mut v) = (vec![0.0f32; 3], vec![0.0f32; 3]);
        let s = AdamWStepScalars::new(0.001, 0.1, 0.9, 0.999, 1e-8, 1);
        adamw_update(&mut w, &g, &mut m, &mut v, &s);
        let decay = 1.0 - 0.001 * 0.1;
        for (got, want) in w
            .iter()
            .zip([decay - 0.001, 2.0 * decay + 0.001, 3.0 * decay - 0.001])
        {
            assert!((got - want).abs() < 1e-6, "{w:?}");
        }
    }

    /// `torch.optim.AdamW` in f64 for one scalar under a constant gradient.
    #[test]
    fn many_steps_follow_a_float64_reference() {
        let (lr, wd, b1, b2, eps) = (1e-3_f64, 0.05, 0.9, 0.999, 1e-8);
        let (mut p, g) = (1.5_f64, -0.4_f64);
        let (mut pm, mut pv) = (0.0, 0.0);
        let mut w = vec![1.5f32];
        let (mut m, mut v) = (vec![0.0f32], vec![0.0f32]);
        for t in 1..=200u64 {
            p *= 1.0 - lr * wd;
            pm = b1 * pm + (1.0 - b1) * g;
            pv = b2 * pv + (1.0 - b2) * g * g;
            p -= lr / (1.0 - b1.powi(t as i32)) * pm
                / (pv.sqrt() / (1.0 - b2.powi(t as i32)).sqrt() + eps);
            let s = AdamWStepScalars::new(lr as f32, wd as f32, b1 as f32, b2 as f32, 1e-8, t);
            adamw_update(&mut w, &[g as f32], &mut m, &mut v, &s);
        }
        assert!((f64::from(w[0]) - p).abs() < 1e-5, "{} vs {p}", w[0]);
    }

    #[test]
    #[should_panic(expected = "1-based")]
    fn step_zero_has_no_bias_correction() {
        let _ = AdamWStepScalars::new(1e-3, 0.0, 0.9, 0.999, 1e-8, 0);
    }
}
