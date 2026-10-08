//! Paired preference objectives: DPO, IPO, hinge (SLiC), SimPO and ORPO.
//!
//! Every objective here scores a (chosen, rejected) pair from the policy's
//! sequence log-probabilities, and those that anchor to a reference model from
//! the reference's as well. They differ in three things, all decided by
//! [`PreferenceLoss`]:
//!
//! - whether a sequence's log-probability is the **sum** over its completion
//!   tokens or the **mean** ([`PreferenceLoss::length_normalized`]);
//! - whether a **reference** model enters ([`PreferenceLoss::uses_reference`]);
//! - the function of the margin that is minimised.
//!
//! The formulas follow the papers and their reference implementations: TRL's
//! `DPOTrainer` for DPO (`sigmoid`, `robust`), IPO and hinge, including IPO's
//! length-averaged log-probs; the SimPO authors' trainer for SimPO; TRL's
//! `ORPOTrainer` for ORPO.

use pmetal_bridge::compat::{Array, Exception, ops};

/// A paired preference objective and its hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreferenceLoss {
    /// Direct Preference Optimization (Rafailov et al. 2023, arXiv:2305.18290).
    ///
    /// `-log σ(β·h)` with `h = (log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))`
    /// on summed log-probs.
    ///
    /// A positive `label_smoothing` ε, the share of preference labels taken to
    /// be flipped, gives Robust DPO (Chowdhury et al. 2024, arXiv:2403.00409):
    /// `(−(1−ε)·log σ(β·h) + ε·log σ(−β·h)) / (1 − 2ε)`, an unbiased estimate
    /// of the clean-label loss (the paper suggests ε = 0.1). It replaces the
    /// earlier conservative-DPO mix, which is biased under label noise.
    Dpo { beta: f32, label_smoothing: f32 },
    /// Identity Preference Optimization (Azar et al. 2023, arXiv:2310.12036).
    ///
    /// `(h − 1/(2β))²` on length-averaged log-probs. Unlike DPO it regresses
    /// the margin to a target rather than pushing it without bound, which
    /// keeps a deterministic preference from driving the policy to extremes.
    Ipo { beta: f32 },
    /// SLiC-style hinge loss (Zhao et al. 2023, arXiv:2305.10425):
    /// `max(0, 1 − β·h)` on summed log-probs.
    Hinge { beta: f32 },
    /// Simple Preference Optimization (Meng et al. 2024, arXiv:2405.14734).
    ///
    /// `−log σ(β·(avg log π(y_w) − avg log π(y_l) − γ/β))`. Reference-free:
    /// the length-averaged log-probability is itself the reward, and γ is the
    /// target margin between the two. As in the reference implementation the
    /// margin is set through `gamma_beta_ratio` (γ/β), which the authors
    /// recommend tuning instead of γ, starting from 0.5.
    Simpo { beta: f32, gamma_beta_ratio: f32 },
    /// Odds Ratio Preference Optimization (Hong et al. 2024, arXiv:2403.07691).
    ///
    /// `NLL(y_w) − β·log σ(log odds(y_w) − log odds(y_l))`, with
    /// `log odds(y) = p − log(1 − e^p)` on the length-averaged log-prob `p`.
    /// Reference-free and single-stage: the NLL term is supervised fine-tuning
    /// on the chosen response.
    Orpo { beta: f32 },
}

/// The loss over a batch and the per-pair implicit rewards behind it.
pub struct PairLoss {
    /// Mean loss over the batch (scalar).
    pub loss: Array,
    /// Implicit reward of each chosen response, `[B]`.
    pub chosen_rewards: Array,
    /// Implicit reward of each rejected response, `[B]`.
    pub rejected_rewards: Array,
}

impl PreferenceLoss {
    /// Name used on the command line and in logs.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dpo { .. } => "dpo",
            Self::Ipo { .. } => "ipo",
            Self::Hinge { .. } => "hinge",
            Self::Simpo { .. } => "simpo",
            Self::Orpo { .. } => "orpo",
        }
    }

    /// The β of this objective.
    pub fn beta(&self) -> f32 {
        match *self {
            Self::Dpo { beta, .. }
            | Self::Ipo { beta }
            | Self::Hinge { beta }
            | Self::Simpo { beta, .. }
            | Self::Orpo { beta } => beta,
        }
    }

    /// Whether sequence log-probs are averaged over completion tokens rather
    /// than summed.
    pub fn length_normalized(&self) -> bool {
        matches!(
            self,
            Self::Ipo { .. } | Self::Simpo { .. } | Self::Orpo { .. }
        )
    }

    /// Whether the objective compares against a frozen reference model.
    pub fn uses_reference(&self) -> bool {
        matches!(
            self,
            Self::Dpo { .. } | Self::Ipo { .. } | Self::Hinge { .. }
        )
    }

    /// Check the hyperparameters.
    pub fn validate(&self) -> Result<(), String> {
        let beta = self.beta();
        if !(beta.is_finite() && beta > 0.0) {
            return Err(format!(
                "{}: beta must be positive, got {beta}",
                self.name()
            ));
        }
        if let Self::Dpo {
            label_smoothing, ..
        } = *self
        {
            if !(0.0..0.5).contains(&label_smoothing) {
                return Err(format!(
                    "dpo: label smoothing must be in [0, 0.5), got {label_smoothing}"
                ));
            }
        }
        if let Self::Simpo {
            gamma_beta_ratio, ..
        } = *self
        {
            if !gamma_beta_ratio.is_finite() {
                return Err(format!(
                    "simpo: gamma/beta ratio must be finite, got {gamma_beta_ratio}"
                ));
            }
        }
        Ok(())
    }

    /// Compute the loss for a batch of pairs.
    ///
    /// `chosen` / `rejected` are the policy's sequence log-probs `[B]`, summed
    /// or averaged as [`Self::length_normalized`] says. `reference` holds the
    /// reference model's, reduced the same way, and must be `Some` exactly when
    /// [`Self::uses_reference`] is true.
    pub fn compute(
        &self,
        chosen: &Array,
        rejected: &Array,
        reference: Option<(&Array, &Array)>,
    ) -> Result<PairLoss, Exception> {
        let (chosen_ratio, rejected_ratio) = match (self.uses_reference(), reference) {
            (true, Some((ref_chosen, ref_rejected))) => {
                (chosen.subtract(ref_chosen), rejected.subtract(ref_rejected))
            }
            (false, None) => (chosen.clone(), rejected.clone()),
            (true, None) => {
                return Err(Exception::custom(format!(
                    "{} needs reference log-probs",
                    self.name()
                )));
            }
            (false, Some(_)) => {
                return Err(Exception::custom(format!(
                    "{} is reference-free; pass no reference log-probs",
                    self.name()
                )));
            }
        };
        let beta = Array::from_f32(self.beta());
        let margin = chosen_ratio.subtract(&rejected_ratio);

        let per_pair = match *self {
            Self::Dpo {
                label_smoothing, ..
            } => {
                let logits = margin.multiply(&beta);
                let loss = softplus(&logits.negative());
                if label_smoothing > 0.0 {
                    // −log σ(−z) = softplus(z), so +ε·log σ(−z) = −ε·softplus(z).
                    loss.multiply(&Array::from_f32(1.0 - label_smoothing))
                        .subtract(&softplus(&logits).multiply(&Array::from_f32(label_smoothing)))
                        .divide(&Array::from_f32(1.0 - 2.0 * label_smoothing))
                } else {
                    loss
                }
            }
            Self::Ipo { beta } => margin
                .subtract(&Array::from_f32(1.0 / (2.0 * beta)))
                .square(),
            Self::Hinge { .. } => ops::maximum(
                &Array::from_f32(1.0).subtract(&margin.multiply(&beta)),
                &Array::from_f32(0.0),
            ),
            Self::Simpo {
                gamma_beta_ratio, ..
            } => softplus(
                &margin
                    .subtract(&Array::from_f32(gamma_beta_ratio))
                    .multiply(&beta)
                    .negative(),
            ),
            Self::Orpo { .. } => {
                let log_odds = margin.subtract(&log1m_exp(chosen).subtract(&log1m_exp(rejected)));
                let ratio_loss = softplus(&log_odds.negative());
                chosen.negative().add(&ratio_loss.multiply(&beta))
            }
        };

        Ok(PairLoss {
            loss: per_pair.mean(None),
            chosen_rewards: chosen_ratio.multiply(&beta),
            rejected_rewards: rejected_ratio.multiply(&beta),
        })
    }
}

/// `log(1 + e^x)`, computed as `logsumexp(x, 0)`.
///
/// The bridge's `softplus` is `log1p(exp(x))`, which is `inf` past x ≈ 88 in
/// f32 and has a NaN gradient there; a pair the policy already ranks very
/// wrongly reaches that range under DPO's summed log-probs. The usual
/// `max(x, 0) + log1p(e^{−|x|})` is no better here: its gradient at exactly
/// 0 comes out 0 rather than ½ (`max` and `abs` have kinks there), and DPO
/// starts every pair at exactly 0, so it would never move. `logsumexp` is
/// stable and its gradient is `σ(x)` everywhere.
pub(crate) fn softplus(x: &Array) -> Array {
    let zeros = ops::zeros_like(x);
    ops::stack_axis(&[x.clone(), zeros], -1).logsumexp(-1, false)
}

/// `log(1 − e^p)` for a log-probability `p ≤ 0`, guarded at `p = 0`.
fn log1m_exp(p: &Array) -> Array {
    let one_minus = Array::from_f32(1.0).subtract(&p.exp());
    ops::maximum(&one_minus, &Array::from_f32(1e-7)).log()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(a: &Array) -> Vec<f32> {
        a.eval();
        pmetal_bridge::check_last_error().expect("bridge error");
        a.as_slice::<f32>().to_vec()
    }

    fn scalar(a: &Array) -> f32 {
        a.eval();
        pmetal_bridge::check_last_error().expect("bridge error");
        a.item_f32()
    }

    fn arr(v: &[f32]) -> Array {
        Array::from_slice(v, &[v.len() as i32])
    }

    fn softplus_f64(x: f64) -> f64 {
        x.max(0.0) + (-x.abs()).exp().ln_1p()
    }

    // Two pairs, hand-computed. Policy and reference sequence log-probs:
    //   pair 0: π(w)=-10, π(l)=-14, ref(w)=-11, ref(l)=-12  → h = 1 - (-2) = 3
    //   pair 1: π(w)=-20, π(l)=-18, ref(w)=-19, ref(l)=-19  → h = -1 - 1  = -2
    const PI_W: [f32; 2] = [-10.0, -20.0];
    const PI_L: [f32; 2] = [-14.0, -18.0];
    const REF_W: [f32; 2] = [-11.0, -19.0];
    const REF_L: [f32; 2] = [-12.0, -19.0];
    const H: [f64; 2] = [3.0, -2.0];

    fn ref_based(loss: PreferenceLoss) -> PairLoss {
        loss.compute(&arr(&PI_W), &arr(&PI_L), Some((&arr(&REF_W), &arr(&REF_L))))
            .unwrap()
    }

    #[test]
    fn dpo_matches_hand_computation() {
        let beta = 0.1;
        let out = ref_based(PreferenceLoss::Dpo {
            beta,
            label_smoothing: 0.0,
        });
        // -log σ(βh) = softplus(-βh)
        let expected = H
            .iter()
            .map(|h| softplus_f64(-(beta as f64) * h))
            .sum::<f64>()
            / 2.0;
        assert!((scalar(&out.loss) as f64 - expected).abs() < 1e-6);
        // rewards = β (π − ref): chosen [0.1, -0.1], rejected [-0.2, 0.1]
        let close =
            |got: Vec<f32>, want: [f32; 2]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6);
        assert!(close(values(&out.chosen_rewards), [0.1, -0.1]));
        assert!(close(values(&out.rejected_rewards), [-0.2, 0.1]));
    }

    /// Robust DPO, as in its paper and TRL's `loss_type="robust"`:
    /// `((1−ε)·softplus(−z) − ε·softplus(z)) / (1 − 2ε)`.
    #[test]
    fn robust_dpo_matches_hand_computation() {
        let (beta, eps) = (0.5f32, 0.1f32);
        let out = ref_based(PreferenceLoss::Dpo {
            beta,
            label_smoothing: eps,
        });
        let eps = eps as f64;
        let expected = H
            .iter()
            .map(|h| {
                let z = beta as f64 * h;
                ((1.0 - eps) * softplus_f64(-z) - eps * softplus_f64(z)) / (1.0 - 2.0 * eps)
            })
            .sum::<f64>()
            / 2.0;
        assert!((scalar(&out.loss) as f64 - expected).abs() < 1e-6);
    }

    #[test]
    fn ipo_regresses_margin_to_half_inverse_beta() {
        let beta = 0.25f32;
        let out = ref_based(PreferenceLoss::Ipo { beta });
        // (h − 1/(2β))², target 2.0: (3−2)² = 1, (−2−2)² = 16 → mean 8.5
        assert!((scalar(&out.loss) - 8.5).abs() < 1e-5);
    }

    #[test]
    fn hinge_is_relu_of_one_minus_beta_h() {
        let out = ref_based(PreferenceLoss::Hinge { beta: 0.5 });
        // max(0, 1 − 1.5) = 0, max(0, 1 + 1) = 2 → mean 1
        assert!((scalar(&out.loss) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn simpo_matches_hand_computation() {
        // Length-averaged log-probs.
        let chosen = arr(&[-0.5, -1.2]);
        let rejected = arr(&[-0.9, -1.0]);
        let (beta, gamma_beta_ratio) = (2.0f32, 0.5f32);
        let out = PreferenceLoss::Simpo {
            beta,
            gamma_beta_ratio,
        }
        .compute(&chosen, &rejected, None)
        .unwrap();
        // β·(Δ − γ/β): pair 0: 2·(0.4 − 0.5) = −0.2; pair 1: 2·(−0.2 − 0.5) = −1.4
        let expected = (softplus_f64(0.2) + softplus_f64(1.4)) / 2.0;
        assert!((scalar(&out.loss) as f64 - expected).abs() < 1e-6);
        let rewards = values(&out.chosen_rewards);
        assert!((rewards[0] + 1.0).abs() < 1e-6 && (rewards[1] + 2.4).abs() < 1e-6);
    }

    #[test]
    fn orpo_matches_hand_computation() {
        let chosen = [-0.5f64, -1.2];
        let rejected = [-0.9f64, -1.0];
        let beta = 0.1f32;
        let out = PreferenceLoss::Orpo { beta }
            .compute(
                &arr(&chosen.map(|x| x as f32)),
                &arr(&rejected.map(|x| x as f32)),
                None,
            )
            .unwrap();
        let log_odds = |p: f64| p - (1.0 - p.exp()).ln();
        let expected = chosen
            .iter()
            .zip(rejected.iter())
            .map(|(&c, &r)| -c + beta as f64 * softplus_f64(-(log_odds(c) - log_odds(r))))
            .sum::<f64>()
            / 2.0;
        assert!((scalar(&out.loss) as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn softplus_is_finite_far_from_zero() {
        let out = values(&softplus(&arr(&[-200.0, -1.0, 0.0, 1.0, 200.0])));
        let expected = [-200.0, -1.0, 0.0, 1.0, 200.0].map(softplus_f64);
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((*a as f64 - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    /// DPO starts every pair at a margin of exactly 0, so the gradient there
    /// must be σ(0) = ½: a softplus with a kink at 0 left DPO stuck at ln 2.
    #[test]
    fn softplus_gradient_is_sigmoid_including_at_zero() {
        let x = arr(&[-3.0, 0.0, 2.0]);
        let (_, grads) = pmetal_bridge::compat::nn::value_and_grad_explicit(
            |p: &[Array]| softplus(&p[0]).sum(None),
            &[x],
            &[],
        )
        .unwrap();
        // Compare on the device: the gradient can come back as a strided view
        // of the stacked cotangent, which `as_slice` would read unstrided.
        let want = arr(&[-3.0f64, 0.0, 2.0].map(|x| (1.0 / (1.0 + (-x).exp())) as f32));
        let err = scalar(&grads[0].subtract(&want).abs().max(None));
        assert!(err < 1e-6, "softplus gradient off sigmoid by {err}");
    }

    #[test]
    fn reference_presence_is_checked() {
        let a = arr(&[-1.0]);
        assert!(
            PreferenceLoss::Dpo {
                beta: 0.1,
                label_smoothing: 0.0
            }
            .compute(&a, &a, None)
            .is_err()
        );
        assert!(
            PreferenceLoss::Orpo { beta: 0.1 }
                .compute(&a, &a, Some((&a, &a)))
                .is_err()
        );
    }

    #[test]
    fn validation_rejects_bad_hyperparameters() {
        assert!(PreferenceLoss::Hinge { beta: 0.0 }.validate().is_err());
        assert!(
            PreferenceLoss::Dpo {
                beta: 0.1,
                label_smoothing: 0.5
            }
            .validate()
            .is_err()
        );
        assert!(
            PreferenceLoss::Simpo {
                beta: 2.0,
                gamma_beta_ratio: 0.5
            }
            .validate()
            .is_ok()
        );
    }
}
