//! GRPO configuration: the objective's variants as explicit options, and the
//! recipes (DAPO, Dr. GRPO, GSPO) that set them together.

use super::{GrpoError, GrpoResult};

/// How the per-token clipped surrogate losses are aggregated into one loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpoLossType {
    /// The original GRPO aggregation (DeepSeekMath, arXiv 2402.03300): each
    /// sequence's tokens averaged over its own length, then the sequences
    /// averaged. It weights a long completion's tokens less than a short
    /// one's, which biases toward short correct and long wrong answers.
    Grpo,
    /// Token-level aggregation (DAPO, arXiv 2503.14476): every token in the
    /// generation batch weighs the same, the sum over the number of
    /// completion tokens.
    #[default]
    Dapo,
    /// Dr. GRPO's aggregation (*Understanding R1-Zero-Like Training*, arXiv
    /// 2503.20783): the sum over tokens divided by a constant, the number of
    /// sequences times `max_completion_length`, so no completion's length
    /// changes how much its tokens count. The paper also drops the
    /// advantages' division by the group's standard deviation;
    /// [`GrpoConfig::with_loss_preset`]`("dr_grpo")` does both.
    DrGrpo,
}

/// The level the policy ratio `π_θ / π_θ_old` is taken at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceSampling {
    /// One ratio per token, as in PPO and GRPO.
    #[default]
    Token,
    /// One ratio per sequence, the length-normalized sequence likelihood ratio
    /// `s_i = exp(mean_t (log π_θ − log π_θ_old))` that GSPO (*Group Sequence
    /// Policy Optimization*, arXiv 2507.18071) clips. It differs from the
    /// token level only once the policy has moved off the one that generated
    /// the batch, i.e. with `num_iterations > 1`.
    Sequence,
}

/// The per-token estimator of `KL(π_θ ‖ π_ref)` the `beta` penalty is built
/// from, differentiated as a loss.
///
/// The estimators agree in value on samples from `π_θ`, but their *gradients*
/// do not: Shah et al. (*A Comedy of Estimators: On KL Regularization in RL
/// Training of LLMs*, arXiv 2512.21852) and Liu et al. (*Rethinking KL
/// Regularization in RLHF*, arXiv 2510.01555) show that `k3` differentiated
/// as a loss follows the gradient of the *forward* KL, not the reverse KL it
/// is meant to penalize, and that `k2` differentiated as a loss is the one
/// whose gradient is the reverse KL's (equivalently, `k1` taken as a reward
/// penalty).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KlEstimator {
    /// `k2 = ½ (log π_θ − log π_ref)²`: its gradient is the reverse KL's.
    #[default]
    K2,
    /// `k3 = π_ref/π_θ − log(π_ref/π_θ) − 1` (Schulman's estimator): never
    /// negative and low-variance as a value, but a biased gradient.
    K3,
}

/// GRPO configuration.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Number of completions to generate per prompt. The advantage is relative
    /// to the group, so at least 2.
    pub num_generations: usize,
    /// Maximum length of the generated completion.
    pub max_completion_length: usize,
    /// Maximum length of the prompt.
    pub max_prompt_length: usize,
    /// KL penalty coefficient. 0 (the default) trains without a reference
    /// model, as DAPO and Dr. GRPO do; above 0 a reference model must be
    /// given to [`GrpoTrainer::train_step`](super::GrpoTrainer::train_step).
    pub beta: f64,
    /// Which estimator the KL penalty uses.
    pub kl_estimator: KlEstimator,
    /// Sampling temperature of the rollouts, which is also the temperature
    /// the policy's log-probs are taken at. Greater than 0: a greedy group is
    /// one completion repeated, with nothing to compare.
    pub temperature: f64,
    /// Top-p of the rollouts. 1 (the default) samples from the full
    /// distribution the log-probs are taken under; truncating it makes the
    /// samples come from a distribution the loss does not describe.
    pub top_p: f64,
    /// Top-k of the rollouts; 0 (the default) disables it. See `top_p`.
    pub top_k: usize,
    /// Min-p of the rollouts; 0 (the default) disables it. See `top_p`.
    pub min_p: f64,
    /// Token ids that end a completion (and are part of it). Empty ends it at
    /// the tokenizer's EOS alone; pass the model's full set (its chat turn's
    /// end among them) so a reply stops where the model would stop.
    pub stop_tokens: Vec<u32>,
    /// Whether to divide advantages by the group's standard deviation
    /// (GRPO) or only subtract its mean (Dr. GRPO).
    pub whiten_advantages: bool,
    /// How token losses are aggregated.
    pub loss_type: GrpoLossType,
    /// Whether the policy ratio is per token or per sequence (GSPO).
    pub importance_sampling: ImportanceSampling,
    /// Optimizer updates per generation batch (μ in the GRPO paper). The
    /// policy ratio is taken against the log-probs of the policy that
    /// generated the batch, so with μ = 1 it is exactly 1 and clipping never
    /// engages; from the second update on it measures how far the policy has
    /// moved, and the clip bounds that.
    pub num_iterations: usize,
    /// Lower clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_low: f64,
    /// Upper clipping epsilon for PPO-clip (default 0.2; DAPO's clip-higher
    /// uses 0.28).
    pub epsilon_high: f64,
    /// Enable VLM (Vision-Language Model) mode for processing image inputs.
    ///
    /// When enabled, the trainer will load images from each prompt's `images`,
    /// pass them to reward functions, and use `forward_with_images` for the training
    /// step to condition the model on visual inputs alongside text.
    pub vlm_mode: bool,
    /// Maximum image size (pixels per side) for VLM preprocessing.
    ///
    /// Images are resized to fit within this square while maintaining aspect ratio.
    /// Typical values: 336 (CLIP ViT-L/14), 448, 560 (Mllama).
    pub max_image_size: usize,
    /// Path to a pretrained ML reward model for scoring completions.
    ///
    /// When set, an `MLRewardModel` is loaded at training start and added to the
    /// `CombinedReward` with weight `reward_model_weight`.  The reward model runs
    /// inference-only alongside the policy model.
    pub reward_model_path: Option<String>,
    /// Maximum input sequence length for the ML reward model (tokens).
    ///
    /// Inputs longer than this are truncated from the right.  Defaults to 2048.
    pub reward_model_max_length: usize,
    /// Weight for the ML reward model relative to heuristic reward functions.
    ///
    /// The combined reward is a weighted sum across all reward functions.
    /// Defaults to 1.0.
    pub reward_model_weight: f64,
    /// Optional chat template for formatting prompt+completion inputs to the
    /// reward model.
    ///
    /// Use `{prompt}` and `{completion}` as placeholders.  When `None`, prompt
    /// and completion are concatenated directly (suitable for reward models
    /// that expect raw text).
    pub reward_model_chat_template: Option<String>,
    /// Enable pipelined (asynchronous) reward scoring.
    ///
    /// When `true`, each step submits the reward scoring request to a
    /// background thread and trains on the batch whose scores arrived: the
    /// scoring of one batch overlaps the generation of the next. The
    /// batch trained on was generated one step earlier, so the policy is one
    /// update ahead of the one that sampled it (the importance ratios are
    /// taken against the policy at training time, so they do not see it).
    ///
    /// Most effective when the reward model is CPU- or ANE-bound; for
    /// heuristic rewards there is nothing to overlap.
    ///
    /// Defaults to `false`.
    pub async_rewards: bool,
    /// KV cache quantization bits for the generation (rollout) phase.
    ///
    /// When `Some(bits)`, the KV cache used during `generate_completions` is
    /// configured with `CacheMode::Quantized` at the requested bit width
    /// (valid values: 2, 4, 8).  A group size of 64 is used, which is
    /// compatible with all standard head dimensions.
    ///
    /// `None` (default) uses the standard fp16 cache.
    pub kv_cache_bits: Option<u8>,
    /// Rollout RNG seed for reproducible generation. Each rollout draws from
    /// a stream derived from it and the rollout's index, so no two rollouts
    /// repeat each other's randomness.
    pub seed: Option<u64>,
    /// DAPO dynamic sampling: drop groups whose completions are all right or
    /// all wrong, since their advantages are all zero.
    pub dapo_dynamic_sampling: bool,
    /// Accuracy threshold margin for DAPO dynamic sampling.
    pub dapo_dynamic_sampling_min_accuracy: f64,
    /// Reward threshold above which a completion is counted as correct.
    pub dapo_accuracy_reward_threshold: f64,
    /// Reward added to completions that stop because they hit the length
    /// limit (DAPO's overlong penalty); `None` leaves their rewards alone.
    pub dapo_overlong_penalty: Option<f64>,
    /// Minimum completions required after DAPO filtering.
    pub dapo_min_group_size: usize,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            max_completion_length: 512,
            max_prompt_length: 512,
            beta: 0.0,
            kl_estimator: KlEstimator::K2,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            stop_tokens: Vec::new(),
            whiten_advantages: true,
            loss_type: GrpoLossType::Dapo,
            importance_sampling: ImportanceSampling::Token,
            num_iterations: 1,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
            vlm_mode: false,
            max_image_size: 336,
            reward_model_path: None,
            reward_model_max_length: 2048,
            reward_model_weight: 1.0,
            reward_model_chat_template: None,
            async_rewards: false,
            kv_cache_bits: None,
            seed: None,
            dapo_dynamic_sampling: false,
            dapo_dynamic_sampling_min_accuracy: 0.01,
            dapo_accuracy_reward_threshold: 0.0,
            dapo_overlong_penalty: None,
            dapo_min_group_size: 2,
        }
    }
}

impl GrpoConfig {
    /// The default configuration with `num_generations` completions a prompt.
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            ..Default::default()
        }
    }

    /// Set the KL penalty coefficient.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// The DAPO recipe (arXiv 2503.14476): token-level loss, no KL term,
    /// clip-higher (ε_low 0.2, ε_high 0.28), dynamic sampling, an overlong
    /// penalty and groups of at least 16.
    pub fn for_dapo(mut self) -> Self {
        self.loss_type = GrpoLossType::Dapo;
        self.beta = 0.0;
        self.epsilon_low = 0.2;
        self.epsilon_high = 0.28;
        self.num_generations = self.num_generations.max(16);
        self.dapo_dynamic_sampling = true;
        self.dapo_overlong_penalty = Some(-1.0);
        self.dapo_min_group_size = 2;
        self
    }

    /// Dr. GRPO (arXiv 2503.20783): the constant-normalized loss and
    /// advantages that are not divided by the group's standard deviation.
    pub fn for_dr_grpo(mut self) -> Self {
        self.loss_type = GrpoLossType::DrGrpo;
        self.whiten_advantages = false;
        self
    }

    /// GSPO (arXiv 2507.18071): the sequence-level ratio, sequences averaged
    /// as in GRPO, and the paper's clip range (ε_low 3e-4, ε_high 4e-4, its
    /// section 5.1). Its ratio only differs from 1 once the policy moves, so
    /// it needs `num_iterations > 1` to do anything.
    pub fn for_gspo(mut self) -> Self {
        self.importance_sampling = ImportanceSampling::Sequence;
        self.loss_type = GrpoLossType::Grpo;
        self.epsilon_low = 3e-4;
        self.epsilon_high = 4e-4;
        self
    }

    /// Apply the method a CLI `--loss-type` names: `dapo` (the default),
    /// `grpo`, `dr_grpo` or `gspo`.
    pub fn with_loss_preset(mut self, name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "dapo" => self.loss_type = GrpoLossType::Dapo,
            "grpo" => self.loss_type = GrpoLossType::Grpo,
            "dr_grpo" => self = self.for_dr_grpo(),
            "gspo" => self = self.for_gspo(),
            other => {
                return Err(format!(
                    "unknown GRPO loss type '{other}'; valid: {}",
                    GRPO_LOSS_PRESETS.join(", ")
                ));
            }
        }
        Ok(self)
    }

    /// Check the configuration, naming every option that cannot work.
    pub fn validate(&self) -> GrpoResult<()> {
        let mut problems: Vec<String> = Vec::new();
        let mut check = |ok: bool, what: String| {
            if !ok {
                problems.push(what);
            }
        };
        check(
            self.num_generations >= 2,
            format!(
                "num_generations is {}; the advantage is relative to the group, so it needs at least 2",
                self.num_generations
            ),
        );
        check(
            self.max_completion_length >= 1,
            "max_completion_length must be at least 1".into(),
        );
        check(
            self.max_prompt_length >= 1,
            "max_prompt_length must be at least 1".into(),
        );
        check(
            self.beta.is_finite() && self.beta >= 0.0,
            format!("beta must be finite and not negative, got {}", self.beta),
        );
        check(
            self.temperature.is_finite() && self.temperature > 0.0,
            format!(
                "temperature must be positive, got {}: a greedy group is one completion repeated",
                self.temperature
            ),
        );
        check(
            self.top_p > 0.0 && self.top_p <= 1.0,
            format!("top_p must be in (0, 1], got {}", self.top_p),
        );
        check(
            (0.0..1.0).contains(&self.min_p),
            format!("min_p must be in [0, 1), got {}", self.min_p),
        );
        check(
            self.num_iterations >= 1,
            "num_iterations must be at least 1".into(),
        );
        check(
            self.epsilon_low.is_finite()
                && self.epsilon_low >= 0.0
                && self.epsilon_high.is_finite()
                && self.epsilon_high >= 0.0,
            format!(
                "the clip epsilons must be finite and not negative, got {} and {}",
                self.epsilon_low, self.epsilon_high
            ),
        );
        if let Some(bits) = self.kv_cache_bits {
            check(
                matches!(bits, 2 | 4 | 8),
                format!("kv_cache_bits must be 2, 4 or 8, got {bits}"),
            );
        }
        check(
            self.dapo_min_group_size >= 1,
            "dapo_min_group_size must be at least 1".into(),
        );
        check(
            (0.0..0.5).contains(&self.dapo_dynamic_sampling_min_accuracy),
            format!(
                "dapo_dynamic_sampling_min_accuracy must be in [0, 0.5), got {}",
                self.dapo_dynamic_sampling_min_accuracy
            ),
        );
        if self.vlm_mode {
            check(
                self.max_image_size >= 1,
                "max_image_size must be at least 1".into(),
            );
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(GrpoError::Config(problems.join("; ")))
        }
    }
}

/// The names [`GrpoConfig::with_loss_preset`] accepts.
pub const GRPO_LOSS_PRESETS: [&str; 4] = ["dapo", "grpo", "dr_grpo", "gspo"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_valid() {
        GrpoConfig::default().validate().unwrap();
        GrpoConfig::default().for_dapo().validate().unwrap();
        GrpoConfig::default().for_dr_grpo().validate().unwrap();
        GrpoConfig::default().for_gspo().validate().unwrap();
    }

    /// Rollouts sample the distribution the log-probs are taken under, and
    /// the KL penalty is off until a reference model is given.
    #[test]
    fn the_defaults_do_not_truncate_the_rollout_distribution() {
        let config = GrpoConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.min_p, 0.0);
        assert_eq!(config.beta, 0.0);
    }

    #[test]
    fn validation_names_every_problem() {
        let config = GrpoConfig {
            num_generations: 1,
            temperature: 0.0,
            top_p: 1.5,
            min_p: 1.0,
            num_iterations: 0,
            beta: f64::NAN,
            kv_cache_bits: Some(3),
            ..GrpoConfig::default()
        };
        let GrpoError::Config(message) = config.validate().unwrap_err() else {
            panic!("expected a configuration error");
        };
        for needle in [
            "num_generations",
            "temperature",
            "top_p",
            "min_p",
            "num_iterations",
            "beta",
            "kv_cache_bits",
        ] {
            assert!(message.contains(needle), "{needle} missing from: {message}");
        }
    }
}
