//! The GRPO objective: group-relative advantages and the clipped surrogate.

use pmetal_runtime::compat::{Array, ops};

use super::config::{GrpoConfig, GrpoLossType, ImportanceSampling, KlEstimator};
use super::{GrpoError, GrpoResult};

/// The parts of one GRPO loss evaluation.
#[derive(Debug, Clone)]
pub struct GrpoLoss {
    /// Policy loss plus `beta` times the KL to the reference: what the
    /// optimizer minimizes.
    pub total: Array,
    /// The clipped surrogate term alone.
    pub policy: Array,
    /// Mean per-token KL to the reference model, as the `k3` estimator
    /// (0 without one): a non-negative, unbiased value whichever estimator
    /// the penalty is differentiated through.
    pub kl: Array,
    /// The importance ratio: `[B, T]` per token, `[B, 1]` per sequence.
    pub ratio: Array,
    /// Share of completion tokens whose objective clipping changed: ratio
    /// below `1 − ε_low` with a negative advantage or above `1 + ε_high` with
    /// a positive one.
    pub clip_fraction: Array,
}

/// Group-relative advantages: each reward minus its group's mean, divided by
/// the group's sample standard deviation when `whiten` (GRPO) and left in the
/// reward's units when not (Dr. GRPO).
///
/// `rewards` holds `num_prompts` groups of equal size, one after another. A
/// group whose rewards are all equal has advantages of exactly 0 and
/// contributes no gradient.
pub fn group_advantages(rewards: &[f64], num_prompts: usize, whiten: bool) -> GrpoResult<Vec<f64>> {
    if num_prompts == 0 {
        return Err(GrpoError::Config("num_prompts must be > 0".into()));
    }
    if rewards.len() % num_prompts != 0 {
        return Err(GrpoError::Config(format!(
            "rewards.len() ({}) must be divisible by num_prompts ({})",
            rewards.len(),
            num_prompts
        )));
    }
    if let Some(bad) = rewards.iter().find(|r| !r.is_finite()) {
        return Err(GrpoError::Reward(format!("a reward is {bad}")));
    }

    let n = rewards.len() / num_prompts;
    if n == 0 {
        return Err(GrpoError::Config("group size must be > 0".into()));
    }

    let mut advantages = vec![0.0; rewards.len()];
    for (group, out) in rewards.chunks(n).zip(advantages.chunks_mut(n)) {
        let mean = group.iter().sum::<f64>() / n as f64;
        let scale = if whiten && n > 1 {
            let variance = group.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            // A group of equal rewards has no spread to divide by, and no
            // advantage to scale.
            variance.sqrt().max(1e-4)
        } else {
            1.0
        };
        for (a, &r) in out.iter_mut().zip(group) {
            *a = (r - mean) / scale;
        }
    }
    Ok(advantages)
}

/// The GRPO loss for one batch of completions.
///
/// The policy term is PPO's clipped surrogate,
/// `−min(w·A, clip(w, 1−ε_low, 1+ε_high)·A)`, with the importance ratio
/// `w` taken against `old_per_token_logps`, the log-probs of the policy
/// that generated the batch: per token, or per sequence as the
/// length-normalized `exp(mean_t log ratio)` (GSPO). It is aggregated as
/// [`GrpoLossType`] says, and the `beta`-weighted KL penalty to the
/// reference ([`KlEstimator`]) is aggregated the same way.
///
/// # Arguments
/// * `per_token_logps` - Current policy per-token log-probs `[B, T]`
/// * `old_per_token_logps` - Generation-time policy per-token log-probs `[B, T]` (detached)
/// * `ref_per_token_logps` - Reference model per-token log-probs `[B, T]` or `None`
/// * `advantages` - Group-normalized advantages `[B]`
/// * `completion_mask` - Valid completion token mask `[B, T]`
pub fn grpo_loss(
    config: &GrpoConfig,
    per_token_logps: &Array,
    old_per_token_logps: &Array,
    ref_per_token_logps: Option<&Array>,
    advantages: &Array,
    completion_mask: &Array,
) -> GrpoLoss {
    let eps_low = config.epsilon_low as f32;
    let eps_high = config.epsilon_high as f32;
    let one = Array::from_f32(1.0);
    let mask = completion_mask;
    let n_seqs = advantages.dim(0);

    let log_ratio = per_token_logps.subtract(old_per_token_logps);
    let log_weights = match config.importance_sampling {
        ImportanceSampling::Token => log_ratio,
        ImportanceSampling::Sequence => {
            let lengths = ops::maximum(&mask.sum_axis(-1, true), &one);
            log_ratio.multiply(mask).sum_axis(-1, true).divide(&lengths)
        }
    };
    let ratio = log_weights.exp();

    // [B] -> [B, 1], broadcasting over tokens.
    let adv = advantages.reshape(&[n_seqs, 1]);
    let lo = Array::from_f32(1.0 - eps_low);
    let hi = Array::from_f32(1.0 + eps_high);
    let clipped = ops::clip(&ratio, Some(&lo), Some(&hi));
    let per_token_policy = ops::minimum(&ratio.multiply(&adv), &clipped.multiply(&adv)).negative();

    let token_count = ops::maximum(&mask.sum(None), &one);
    let reduce = |per_token: &Array| -> Array {
        let masked = per_token.multiply(mask);
        match config.loss_type {
            GrpoLossType::Grpo => {
                let lengths = ops::maximum(&mask.sum_axis(-1, false), &one);
                masked.sum_axis(-1, false).divide(&lengths).mean(None)
            }
            GrpoLossType::Dapo => masked.sum(None).divide(&token_count),
            GrpoLossType::DrGrpo => {
                let budget = (n_seqs as usize * config.max_completion_length.max(1)) as f32;
                masked.sum(None).divide(&Array::from_f32(budget))
            }
        }
    };
    let policy = reduce(&per_token_policy);

    let (kl, total) = match ref_per_token_logps {
        Some(ref_logps) => {
            let d = ref_logps.subtract(per_token_logps);
            let k3 = d.exp().subtract(&one).subtract(&d);
            let kl = k3.multiply(mask).sum(None).divide(&token_count);
            let total = if config.beta != 0.0 {
                let penalty = match config.kl_estimator {
                    KlEstimator::K3 => k3,
                    KlEstimator::K2 => d.square().multiply(&Array::from_f32(0.5)),
                };
                policy.add(&reduce(&penalty).multiply(&Array::from_f32(config.beta as f32)))
            } else {
                policy.clone()
            };
            (kl, total)
        }
        None => (Array::from_f32(0.0), policy.clone()),
    };

    // Where clipping changed the objective (no gradient flows through it).
    let w = ratio.stop_gradient();
    let f32_dtype = pmetal_runtime::dtype::F32;
    let zero = Array::from_f32(0.0);
    let low = w
        .less(&lo)
        .as_dtype(f32_dtype)
        .multiply(&adv.less(&zero).as_dtype(f32_dtype));
    let high = w
        .greater(&hi)
        .as_dtype(f32_dtype)
        .multiply(&adv.greater(&zero).as_dtype(f32_dtype));
    let clip_fraction = low.add(&high).multiply(mask).sum(None).divide(&token_count);

    GrpoLoss {
        total,
        policy,
        kl,
        ratio,
        clip_fraction,
    }
}
