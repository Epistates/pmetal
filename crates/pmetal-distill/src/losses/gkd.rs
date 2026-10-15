//! Generalized Knowledge Distillation (GKD): on-policy distillation.
//!
//! Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist and Bachem, *On-Policy
//! Distillation of Language Models: Learning from Self-Generated Mistakes*
//! (ICLR 2024). Distilling on a fixed corpus trains the student on sequences
//! it would never produce itself, so its own mistakes are never corrected.
//! GKD trains on the student's own samples scored by the teacher, and treats
//! the data source and the divergence as free choices:
//!
//! * **data**: with probability `λ` a batch is *on-policy* (the student
//!   generates the responses), otherwise it comes from the fixed corpus (the
//!   paper also admits teacher-generated data);
//! * **divergence**: any token-level divergence of the teacher from the
//!   student: forward KL, reverse KL, or the generalized JSD `JSD(β)`
//!   ([`JsdSkewedLoss`](super::JsdSkewedLoss)); the paper finds the best
//!   choice depends on the task.
//!
//! The paper's `λ` is a *sampling probability* (use
//! [`GkdLoss::use_student_data`] to flip the coin per batch). This type also
//! offers [`compute_full`](GkdLoss::compute_full), the expectation of that
//! coin: a `(1-λ)`/`λ` blend of the divergence on an off-policy pair and an
//! on-policy pair, for loops that have both in hand.
//!
//! The rollout (generating from the student) is the training loop's job; the
//! [`OnPolicySampler`] trait is the seam, with [`GreedySampler`] and
//! [`TemperatureSampler`] as implementations.

use super::{DistillLoss, KlDivergenceLoss};
use crate::Result;
use pmetal_runtime::compat::{Array, Dtype, random};

/// Trait for an on-policy sampler. The trainer implements this to produce
/// student-generated sequences that the teacher then scores. Stays simple
/// on purpose — a richer interface (KV-cache reuse, batched draws) belongs
/// in `pmetal-trainer`.
pub trait OnPolicySampler: Send + Sync {
    /// Sample a single token id from the student's per-position logits.
    ///
    /// `student_logits` has shape `[batch, seq, vocab]`. Returns an `[batch, seq]`
    /// array of int32 token ids — one per position.
    fn sample(&self, student_logits: &Array) -> Result<Array>;
}

/// Greedy "sampler" that always picks the argmax. Useful for tests and as a
/// reference: GKD with greedy sampling reduces to off-policy KD with
/// teacher-forced sequences passed through the student's argmax map.
pub struct GreedySampler;

impl OnPolicySampler for GreedySampler {
    fn sample(&self, student_logits: &Array) -> Result<Array> {
        Ok(student_logits.argmax(-1))
    }
}

/// Samples from `softmax(logits / temperature)`, optionally restricted to the
/// `top_k` most likely tokens.
///
/// Draws come from the thread's global MLX key, so [`random::seed`] makes a
/// run reproducible.
#[derive(Debug, Clone, Copy)]
pub struct TemperatureSampler {
    temperature: f32,
    top_k: Option<usize>,
}

impl TemperatureSampler {
    /// A sampler at `temperature` (clamped to at least `1e-6`; the paper
    /// samples at `T = 1`).
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature: temperature.max(1e-6),
            top_k: None,
        }
    }

    /// Keep only the `k` most likely tokens before sampling.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k.max(1));
        self
    }
}

impl OnPolicySampler for TemperatureSampler {
    fn sample(&self, student_logits: &Array) -> Result<Array> {
        let scaled = student_logits
            .as_dtype(Dtype::Float32.as_i32())
            .divide(&Array::from_f32(self.temperature));
        let scaled = match self.top_k {
            Some(k) if (k as i32) < scaled.dim(-1) => {
                // Everything below the k-th largest logit is out. A tie at the
                // threshold keeps both tied tokens.
                let kth = scaled.topk(k as i32, -1).min_axis(-1, true);
                let keep = scaled.greater_equal(&kth);
                keep.where_cond(&scaled, &Array::from_f32(f32::NEG_INFINITY))
            }
            _ => scaled,
        };
        Ok(random::categorical(&scaled, -1, None))
    }
}

/// GKD loss with a configurable on-policy probability `lambda` and divergence.
pub struct GkdLoss {
    lambda: f32,
    sampler_temperature: f32,
    /// The token-level divergence applied to both the off-policy and the
    /// on-policy pair. Defaults to forward KL.
    inner: Box<dyn DistillLoss>,
}

impl GkdLoss {
    /// `lambda ∈ [0, 1]` controls the on-policy weight. `λ = 0` is plain
    /// off-policy KD; `λ = 1` is fully on-policy. Defaults to forward KL as
    /// the divergence.
    pub fn new(lambda: f32, sampler_temperature: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
            sampler_temperature: sampler_temperature.max(1e-6),
            inner: Box::new(KlDivergenceLoss::new()),
        }
    }

    /// Set the cross-vocab top-k for the default forward-KL divergence. Has
    /// no effect after [`with_divergence`](Self::with_divergence), which
    /// brings its own.
    pub fn with_sparse_top_k(mut self, k: i32) -> Self {
        self.inner = Box::new(KlDivergenceLoss::new().with_sparse_top_k(k.max(1)));
        self
    }

    /// Use another divergence: `JsdSkewedLoss::new(0.1)` for the paper's
    /// `JSD(0.1)`, `KlDivergenceLoss::reverse()` for reverse KL, a
    /// [`SkewKlLoss`](super::SkewKlLoss), ...
    pub fn with_divergence(mut self, divergence: Box<dyn DistillLoss>) -> Self {
        self.inner = divergence;
        self
    }

    /// `lambda` actually in use after clamping.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Sampler temperature (only used when the trainer constructs an
    /// auxiliary `OnPolicySampler` from this configuration).
    pub fn sampler_temperature(&self) -> f32 {
        self.sampler_temperature
    }

    /// A [`TemperatureSampler`] at this loss's sampler temperature.
    pub fn sampler(&self) -> TemperatureSampler {
        TemperatureSampler::new(self.sampler_temperature)
    }

    /// The paper's per-batch coin: whether this batch should be generated by
    /// the student. `uniform` is a draw from `[0, 1)`; the batch is on-policy
    /// when it falls below `λ`.
    pub fn use_student_data(&self, uniform: f32) -> bool {
        uniform < self.lambda
    }

    /// The `λ`-weighted blend of the divergence on an off-policy pair
    /// (fixed-corpus sequences) and an on-policy pair (student-generated
    /// sequences): `(1-λ)·D_off + λ·D_on`, the expectation of the paper's
    /// per-batch coin.
    ///
    /// `temperature` softens both divergences. Use
    /// [`compute_full_weighted`](Self::compute_full_weighted) to mask padding
    /// and prompt positions.
    pub fn compute_full(
        &self,
        teacher_off_logits: &Array,
        student_off_logits: &Array,
        teacher_on_logits: &Array,
        student_on_logits: &Array,
        temperature: f32,
    ) -> Result<Array> {
        self.compute_full_weighted(
            (teacher_off_logits, student_off_logits, None),
            (teacher_on_logits, student_on_logits, None),
            temperature,
        )
    }

    /// [`compute_full`](Self::compute_full) with a per-token weight (a 0/1
    /// mask of the positions that count) on each pair. Each pair is a
    /// `(teacher_logits, student_logits, weights)` triple.
    pub fn compute_full_weighted(
        &self,
        off_policy: (&Array, &Array, Option<&Array>),
        on_policy: (&Array, &Array, Option<&Array>),
        temperature: f32,
    ) -> Result<Array> {
        let off_loss =
            self.inner
                .compute_weighted(off_policy.0, off_policy.1, temperature, off_policy.2)?;
        let on_loss =
            self.inner
                .compute_weighted(on_policy.0, on_policy.1, temperature, on_policy.2)?;
        let off_w = Array::from_f32(1.0 - self.lambda);
        let on_w = Array::from_f32(self.lambda);
        Ok(off_loss.multiply(&off_w).add(&on_loss.multiply(&on_w)))
    }
}

impl DistillLoss for GkdLoss {
    /// Loss-only path: with no on-policy sequence available we cannot
    /// compute the GKD blend, so we fall through to the divergence on
    /// the off-policy pair the caller did supply. The trainer should
    /// call [`GkdLoss::compute_full`] directly to get the actual
    /// `λ`-weighted blend.
    ///
    /// We deliberately do **not** scale by `(1 - λ)` here: at `λ = 1.0`
    /// that would zero out the gradient and silently kill training,
    /// which is a worse failure mode than "loss magnitude doesn't track
    /// the eventual blend". Callers that want the blended magnitude
    /// must use `compute_full`.
    fn weighted_loss(
        &self,
        teacher_logits: &Array,
        student_logits: &Array,
        temperature: f32,
        weights: Option<&Array>,
    ) -> Result<Array> {
        self.inner
            .compute_weighted(teacher_logits, student_logits, temperature, weights)
    }

    fn temperature_scale(&self, temperature: f32) -> f32 {
        self.inner.temperature_scale(temperature)
    }

    fn name(&self) -> &'static str {
        "gkd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::losses::JsdSkewedLoss;
    use serial_test::serial;

    /// `λ = 0` reduces to off-policy KL exactly.
    #[test]
    #[serial]
    fn lambda_zero_matches_off_policy_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let on_t = teacher.clone();
        let on_s = student.clone();

        let gkd = GkdLoss::new(0.0, 1.0);
        let blend: f32 = gkd
            .compute_full(&teacher, &student, &on_t, &on_s, 1.0)
            .unwrap()
            .item();
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        assert!(
            (blend - kl).abs() < 1e-3,
            "λ=0 must reduce to KL: {} vs {}",
            blend,
            kl
        );
    }

    /// Identical off and on logit pairs: blended loss equals KL regardless
    /// of `λ`.
    #[test]
    #[serial]
    fn identical_streams_collapse_to_kl() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        for &lambda in &[0.1_f32, 0.5, 0.9] {
            let v: f32 = GkdLoss::new(lambda, 1.0)
                .compute_full(&teacher, &student, &teacher, &student, 1.0)
                .unwrap()
                .item();
            assert!(
                (v - kl).abs() < 1e-3,
                "λ={} should collapse: {} vs {}",
                lambda,
                v,
                kl
            );
        }
    }

    /// The blend is `(1-λ)·D_off + λ·D_on`, with the padding of each pair
    /// masked by its own weights and the divergence the caller chose.
    #[test]
    #[serial]
    fn weighted_blend_uses_each_pairs_mask_and_the_chosen_divergence() {
        let t_off = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 0.5, 1.5, 2.5], &[1, 2, 3]);
        let s_off = Array::from_f32_slice(&[3.0_f32, 1.0, 0.5, 2.0, 1.0, 0.0], &[1, 2, 3]);
        let t_on = Array::from_f32_slice(&[0.0_f32, 2.0, 1.0, 2.5, 0.5, 1.5], &[1, 2, 3]);
        let s_on = Array::from_f32_slice(&[1.0_f32, 0.0, 2.0, 0.5, 2.0, 1.0], &[1, 2, 3]);
        let off_mask = Array::from_f32_slice(&[1.0_f32, 0.0], &[1, 2]);
        let on_mask = Array::from_f32_slice(&[0.0_f32, 1.0], &[1, 2]);

        let lambda = 0.25_f32;
        let divergence = JsdSkewedLoss::new(0.1);
        let off: f32 = divergence
            .compute_masked(&t_off, &s_off, 1.0, &off_mask)
            .unwrap()
            .item();
        let on: f32 = divergence
            .compute_masked(&t_on, &s_on, 1.0, &on_mask)
            .unwrap()
            .item();

        let gkd = GkdLoss::new(lambda, 1.0).with_divergence(Box::new(JsdSkewedLoss::new(0.1)));
        let blend: f32 = gkd
            .compute_full_weighted(
                (&t_off, &s_off, Some(&off_mask)),
                (&t_on, &s_on, Some(&on_mask)),
                1.0,
            )
            .unwrap()
            .item();
        let want = (1.0 - lambda) * off + lambda * on;
        assert!((blend - want).abs() < 1e-5, "{blend} vs {want}");
    }

    /// The per-batch coin comes up on-policy with probability λ.
    #[test]
    fn the_coin_is_on_policy_below_lambda() {
        let gkd = GkdLoss::new(0.3, 1.0);
        assert!(gkd.use_student_data(0.0));
        assert!(gkd.use_student_data(0.299));
        assert!(!gkd.use_student_data(0.3));
        assert!(!gkd.use_student_data(0.99));
        assert!(!GkdLoss::new(0.0, 1.0).use_student_data(0.0));
    }

    /// Greedy sampler returns argmax shape `[batch, seq]`. Materializes
    /// through f32 since the runtime does not currently expose a typed
    /// int-extract helper for i32 arrays.
    #[test]
    #[serial]
    fn greedy_sampler_returns_argmax() {
        let logits =
            Array::from_f32_slice(&[0.0_f32, 1.0, 5.0, 2.0, 3.0, 4.0, 0.0, 0.0], &[1, 2, 4]);
        let sampled = GreedySampler.sample(&logits).unwrap();
        assert_eq!(sampled.shape(), &[1, 2]);
        let mut as_f = sampled.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
        let ids: Vec<f32> = as_f.to_f32_vec(2).unwrap();
        assert_eq!(ids, vec![2.0_f32, 1.0]);
    }

    /// The temperature sampler draws from `softmax(logits / T)`: over many
    /// draws the empirical frequencies match it, and `top_k` removes the tail.
    #[test]
    #[serial]
    fn temperature_sampler_draws_from_the_tempered_softmax() {
        let n = 20_000_i32;
        let row = [2.0_f32, 1.0, 0.0, -1.0];
        let data: Vec<f32> = (0..n).flat_map(|_| row).collect();
        let logits = Array::from_f32_slice(&data, &[1, n, 4]);

        let frequencies = |sampler: &TemperatureSampler| -> [f64; 4] {
            random::seed(7);
            let mut ids = sampler
                .sample(&logits)
                .unwrap()
                .as_dtype(Dtype::Float32.as_i32());
            let ids = ids.to_f32_vec(n as usize).unwrap();
            let mut counts = [0.0_f64; 4];
            for id in ids {
                counts[id as usize] += 1.0;
            }
            counts.map(|c| c / n as f64)
        };

        for temperature in [1.0_f32, 2.0] {
            let freq = frequencies(&TemperatureSampler::new(temperature));
            let z: f64 = row
                .iter()
                .map(|x| (*x as f64 / temperature as f64).exp())
                .sum();
            for (i, x) in row.iter().enumerate() {
                let p = (*x as f64 / temperature as f64).exp() / z;
                // Five standard errors of a binomial proportion.
                let tol = 5.0 * (p * (1.0 - p) / n as f64).sqrt();
                assert!(
                    (freq[i] - p).abs() <= tol,
                    "T={temperature} token {i}: frequency {} vs probability {p}",
                    freq[i]
                );
            }
        }

        let top2 = frequencies(&TemperatureSampler::new(1.0).with_top_k(2));
        assert_eq!(top2[2] + top2[3], 0.0, "top-2 sampled outside the top 2");
        // Two tokens left, with logits 2 and 1: e^2 / (e^2 + e^1).
        let p0 = 1.0 / (1.0 + (-1.0_f64).exp());
        assert!((top2[0] - p0).abs() < 0.02, "{top2:?}");
    }

    /// `DistillLoss` path: when no on-policy sequence is provided, the
    /// off-only loss is plain off-policy KL — independent of `λ`. This
    /// is a deliberate contract: scaling by `(1-λ)` would silently zero
    /// out training at `λ=1`, which is a far worse failure mode than
    /// "loss magnitude doesn't track the eventual blend".
    #[test]
    #[serial]
    fn off_only_compute_is_plain_kl_regardless_of_lambda() {
        let teacher = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let student = Array::from_f32_slice(&[3.0_f32, 1.0, 4.0, 1.0], &[1, 1, 4]);
        let kl: f32 = KlDivergenceLoss::new()
            .compute(&teacher, &student, 1.0)
            .unwrap()
            .item();
        for &lambda in &[0.0_f32, 0.5, 1.0] {
            let v: f32 = GkdLoss::new(lambda, 1.0)
                .compute(&teacher, &student, 1.0)
                .unwrap()
                .item();
            assert!(
                (v - kl).abs() < 1e-3,
                "off-only path must equal plain KL at λ={}, got {} vs {}",
                lambda,
                v,
                kl
            );
        }
    }
}
