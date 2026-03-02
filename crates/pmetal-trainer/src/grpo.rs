//! Group Relative Policy Optimization (GRPO) implementation.
//!
//! GRPO is a reinforcement learning algorithm that optimizes policies by comparing
//! the performance of multiple completions for the same prompt.
//! It is particularly effective for reasoning models (e.g., DeepSeek-R1).
//!
//! ## Key Features
//! - **Reference-free or Reference-based**: Supports KL divergence from a reference model.
//! - **Group-based Advantages**: Computes advantages relative to the group mean/std.
//! - **Flexible Rewards**: Pluggable reward functions for reasoning, formatting, and accuracy.
//! - **Efficient Training**: Implementation optimized for Apple Silicon via MLX.

use mlx_rs::{
    Array,
    error::Exception,
    module::{Module, ModuleParameters},
    nn,
    ops::indexing::IndexOp,
    optimizers::Optimizer,
};
use pmetal_core::TrainingConfig;
use pmetal_lora::TrainableModel;
use pmetal_models::rl_generation::{BatchedRlConfig, BatchedRlGenerator};
use std::time::Instant;
use tracing::info;

/// Iteration statistics for GRPO training.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GrpoIterationStats {
    /// Training step.
    pub step: usize,
    /// Total loss.
    pub loss: f32,
    /// KL divergence between policy and reference.
    pub kl: f32,
    /// Policy gradient loss.
    pub policy_loss: f32,
    /// Mean reward for this batch.
    pub reward: f32,
    /// Mean advantage for this batch.
    pub advantage: f32,
    /// Generation throughput (completions/sec).
    pub completions_per_second: f32,
}

/// Error type for GRPO training.
#[derive(Debug, thiserror::Error)]
pub enum GrpoError {
    /// MLX error.
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Generation error.
    #[error("Generation error: {0}")]
    Generation(String),
    /// Reward computation error.
    #[error("Reward error: {0}")]
    Reward(String),
    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

/// Result type for GRPO operations.
pub type GrpoResult<T> = std::result::Result<T, GrpoError>;

/// GRPO loss type variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrpoLossType {
    /// Standard GRPO / BNPO (Base Normalized Policy Optimization).
    #[default]
    Bnpo,
    /// DR-GRPO (Detailed Reward GRPO).
    DrGrpo,
    /// DAPO (Distribution-Aware Policy Optimization).
    Dapo,
    /// Simple REINFORCE-style loss without KL.
    Reinforce,
}

/// GRPO configuration.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Number of completions to generate per prompt.
    pub num_generations: usize,
    /// Maximum length of the generated completion.
    pub max_completion_length: usize,
    /// Maximum length of the prompt.
    pub max_prompt_length: usize,
    /// KL divergence coefficient.
    pub beta: f64,
    /// Temperature for sampling.
    pub temperature: f64,
    /// Top-p for sampling.
    pub top_p: f64,
    /// Top-k for sampling.
    pub top_k: usize,
    /// Whether to whiten (normalize) advantages within each group.
    pub whiten_advantages: bool,
    /// Entropy bonus coefficient.
    pub entropy_coef: f64,
    /// Loss function type.
    pub loss_type: GrpoLossType,
    /// Lower clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_low: f64,
    /// Upper clipping epsilon for PPO-clip (default 0.2).
    pub epsilon_high: f64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            max_completion_length: 512,
            max_prompt_length: 512,
            beta: 0.1,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 40,
            whiten_advantages: true,
            entropy_coef: 0.0,
            loss_type: GrpoLossType::Bnpo,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
        }
    }
}

impl GrpoConfig {
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            ..Default::default()
        }
    }

    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    pub fn for_dapo(mut self) -> Self {
        self.loss_type = GrpoLossType::Dapo;
        self.beta = 0.0; // DAPO usually doesn't use standard KL
        self
    }
}

/// Completion group for a single prompt.
#[derive(Debug, Clone)]
pub struct CompletionGroup {
    pub prompt_ids: Vec<u32>,
    pub completion_ids: Vec<Vec<u32>>,
    pub rewards: Vec<f64>,
    pub stopped_by_length: Vec<bool>,
}

impl CompletionGroup {
    pub fn new(prompt_ids: Vec<u32>, num_generations: usize) -> Self {
        Self {
            prompt_ids,
            completion_ids: Vec::with_capacity(num_generations),
            rewards: Vec::with_capacity(num_generations),
            stopped_by_length: Vec::with_capacity(num_generations),
        }
    }

    pub fn add_completion(&mut self, ids: Vec<u32>, reward: f64, stopped_by_length: bool) {
        self.completion_ids.push(ids);
        self.rewards.push(reward);
        self.stopped_by_length.push(stopped_by_length);
    }
}

/// GRPO Trainer.
pub struct GrpoTrainer {
    pub config: GrpoConfig,
    pub training_config: TrainingConfig,
    pub step: usize,
}

impl GrpoTrainer {
    pub fn new(config: GrpoConfig, training_config: TrainingConfig) -> GrpoResult<Self> {
        Ok(Self {
            config,
            training_config,
            step: 0,
        })
    }

    /// Compute per-token log probabilities for a sequence.
    ///
    /// Uses `selective_log_softmax` to avoid materializing the full `[B, S, V]`
    /// log_softmax tensor (~4 GB for 128K-vocab models at typical batch sizes).
    ///
    /// Returns `(per_token_logps, completion_mask)` both `[B, T]` where `T = seq_len - 1`
    /// (shifted for next-token prediction). The mask is 1.0 for valid completion
    /// tokens and 0.0 for prompt/padding tokens.
    pub fn compute_per_token_logps(
        &self,
        logits: &Array,
        labels: &Array,
        temperature: Option<f32>,
    ) -> GrpoResult<(Array, Array)> {
        let l = logits.dim(1);

        // Shift logits and labels for next-token prediction
        let shift_logits = logits.index((.., ..l - 1, ..));
        let shift_labels = labels.index((.., 1..));

        // Memory-efficient: gathers single logit per position via take_along_axis,
        // never materializes [B, S, V] log_softmax.
        let (per_token_logps, valid_mask) =
            crate::logprob_utils::selective_log_softmax_with_temperature(
                &shift_logits,
                &shift_labels,
                temperature,
            )?;

        Ok((per_token_logps, valid_mask))
    }

    /// Compute advantages using group-relative normalization.
    pub fn compute_advantages(&self, rewards: &[f64], num_prompts: usize) -> GrpoResult<Vec<f64>> {
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

        let n_per_group = rewards.len() / num_prompts;
        if n_per_group == 0 {
            return Err(GrpoError::Config("group size must be > 0".into()));
        }

        let mut advantages = vec![0.0; rewards.len()];

        for i in 0..num_prompts {
            let group = &rewards[i * n_per_group..(i + 1) * n_per_group];
            let mean = group.iter().sum::<f64>() / n_per_group as f64;

            if self.config.whiten_advantages && n_per_group > 1 {
                // Normalize by group std (whitening)
                let variance = group.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
                    / (n_per_group - 1) as f64;
                let std = variance.sqrt().max(1e-4);
                for j in 0..n_per_group {
                    advantages[i * n_per_group + j] = (group[j] - mean) / std;
                }
            } else {
                // Raw advantages (reward - mean) without normalization
                for j in 0..n_per_group {
                    advantages[i * n_per_group + j] = group[j] - mean;
                }
            }
        }

        Ok(advantages)
    }

    /// Compute GRPO loss components at the per-token level.
    ///
    /// All variants use PPO-clip as the policy gradient objective (matching TRL):
    /// `L_policy = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)`
    ///
    /// The ratio is computed against `old_per_token_logps` (generation-time policy),
    /// NOT the reference model. KL regularization uses the reference model separately.
    ///
    /// # Arguments
    /// * `per_token_logps` - Current policy per-token log-probs `[B, T]`
    /// * `old_per_token_logps` - Generation-time policy per-token log-probs `[B, T]` (detached)
    /// * `ref_per_token_logps` - Reference model per-token log-probs `[B, T]` or `None`
    /// * `advantages` - Group-normalized advantages `[B]`
    /// * `completion_mask` - Valid completion token mask `[B, T]`
    /// * `entropy` - Optional per-token entropy for bonus
    pub fn compute_grpo_loss(
        &self,
        per_token_logps: &Array,
        old_per_token_logps: &Array,
        ref_per_token_logps: Option<&Array>,
        advantages: &Array,
        completion_mask: &Array,
        entropy: Option<&Array>,
    ) -> GrpoResult<(Array, Array, Array)> {
        let eps_low = self.config.epsilon_low as f32;
        let eps_high = self.config.epsilon_high as f32;

        // --- PPO-clip policy loss (all variants) ---
        // Importance ratio against OLD policy (generation-time), not reference
        let log_ratio = per_token_logps.subtract(old_per_token_logps)?;
        let ratio = log_ratio.exp()?;

        // Expand advantages for token-level broadcasting: [B] -> [B, 1]
        let adv_expanded = advantages.reshape(&[advantages.dim(0), 1])?;

        // Clipped surrogate objective
        let clipped_ratio = mlx_rs::ops::clip(
            &ratio,
            (
                &Array::from_f32(1.0 - eps_low),
                &Array::from_f32(1.0 + eps_high),
            ),
        )?;
        let surr1 = ratio.multiply(&adv_expanded)?;
        let surr2 = clipped_ratio.multiply(&adv_expanded)?;
        let token_policy_loss = mlx_rs::ops::minimum(&surr1, &surr2)?.negative()?;

        // --- Per-variant reduction ---
        let masked_policy_loss = token_policy_loss.multiply(completion_mask)?;
        let total_tokens = completion_mask.sum(None)?;
        let safe_token_count = mlx_rs::ops::maximum(&total_tokens, &Array::from_f32(1.0))?;

        let policy_loss = match self.config.loss_type {
            GrpoLossType::Bnpo => {
                // BNPO: mean over valid tokens
                masked_policy_loss.sum(None)?.divide(&safe_token_count)?
            }
            GrpoLossType::DrGrpo => {
                // DR-GRPO: per-sequence mean, then batch mean
                // Sum tokens per sequence, divide by per-sequence token count
                let per_seq_sum = masked_policy_loss.sum_axis(-1, false)?;
                let per_seq_count = completion_mask.sum_axis(-1, false)?;
                let safe_per_seq = mlx_rs::ops::maximum(&per_seq_count, &Array::from_f32(1.0))?;
                per_seq_sum.divide(&safe_per_seq)?.mean(None)?
            }
            GrpoLossType::Dapo => {
                // DAPO: token-level mean (same as BNPO but conceptually distinct)
                masked_policy_loss.sum(None)?.divide(&safe_token_count)?
            }
            GrpoLossType::Reinforce => {
                // REINFORCE: simple batch mean
                masked_policy_loss.sum(None)?.divide(&safe_token_count)?
            }
        };

        // --- KL divergence (against reference model, not old policy) ---
        // KL(pi || ref) ≈ exp(ref - pi) - (ref - pi) - 1  (Schulman approximation)
        // Correct direction: ratio = ref/pi, KL = ratio - 1 - log(ratio)
        let kl_mean = if let Some(ref_logps) = ref_per_token_logps {
            let kl_log_ratio = ref_logps.subtract(per_token_logps)?;
            let kl_ratio = kl_log_ratio.exp()?;
            let per_token_kl = kl_ratio
                .subtract(&Array::from_f32(1.0))?
                .subtract(&kl_log_ratio)?;
            let masked_kl = per_token_kl.multiply(completion_mask)?;
            masked_kl.sum(None)?.divide(&safe_token_count)?
        } else {
            Array::from_f32(0.0)
        };

        let kl_loss = kl_mean.multiply(&Array::from_f32(self.config.beta as f32))?;
        let mut total_loss = policy_loss.add(&kl_loss)?;

        // Entropy bonus: subtract entropy_coef * entropy to encourage exploration
        if let (Some(ent), coef) = (entropy, self.config.entropy_coef) {
            if coef > 0.0 {
                let masked_ent = ent.multiply(completion_mask)?;
                let mean_ent = masked_ent.sum(None)?.divide(&safe_token_count)?;
                let entropy_bonus = mean_ent.multiply(&Array::from_f32(coef as f32))?;
                total_loss = total_loss.subtract(&entropy_bonus)?;
            }
        }

        Ok((total_loss, kl_mean, policy_loss))
    }

    /// Prepare a training batch from completion groups.
    pub fn prepare_batch(
        &mut self,
        groups: &[CompletionGroup],
    ) -> GrpoResult<(
        Vec<Vec<u32>>,
        Vec<Vec<u32>>,
        Vec<f64>,
        Vec<Vec<f32>>,
        Option<Vec<Vec<Array>>>,
    )> {
        let mut all_prompts = Vec::new();
        let mut all_completions = Vec::new();
        let mut all_rewards = Vec::new();
        let mut all_masks = Vec::new();

        for group in groups {
            for completion in &group.completion_ids {
                all_prompts.push(group.prompt_ids.clone());
                all_completions.push(completion.clone());

                let mut mask = vec![0.0f32; group.prompt_ids.len()];
                mask.extend(vec![1.0f32; completion.len()]);
                all_masks.push(mask);
            }
            all_rewards.extend(&group.rewards);
        }

        let advantages = self.compute_advantages(&all_rewards, groups.len())?;

        Ok((all_prompts, all_completions, advantages, all_masks, None))
    }

    /// Run a single training step on a batch of completion groups.
    ///
    /// Implements the correct GRPO training loop matching TRL:
    /// 1. Build padded input_ids / labels / completion_mask tensors
    /// 2. Compute `old_per_token_logps` from the CURRENT policy (generation-time snapshot)
    /// 3. Optionally compute `ref_per_token_logps` from the reference model
    /// 4. Run `value_and_grad` with `compute_grpo_loss` (PPO-clip on old, KL on ref)
    /// 5. Update optimizer
    pub fn train_step<M, R, O>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        groups: &[CompletionGroup],
        optimizer: &mut O,
    ) -> GrpoResult<GrpoIterationStats>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        let start_time = Instant::now();
        let (all_prompts, all_completions, advantages, _all_masks, _) =
            self.prepare_batch(groups)?;

        // Collect raw rewards for logging before they're normalized into advantages
        let raw_rewards: Vec<f64> = groups
            .iter()
            .flat_map(|g| g.rewards.iter().copied())
            .collect();

        let n_completions = all_completions.len();
        let adv_array = Array::from_slice(
            &advantages.iter().map(|&a| a as f32).collect::<Vec<_>>(),
            &[n_completions as i32],
        );

        let max_len = all_prompts
            .iter()
            .zip(all_completions.iter())
            .map(|(p, c)| p.len() + c.len())
            .max()
            .unwrap_or(0);

        let mut input_ids_vec = Vec::with_capacity(n_completions * max_len);
        let mut labels_vec = Vec::with_capacity(n_completions * max_len);

        for (p, c) in all_prompts.iter().zip(all_completions.iter()) {
            let mut ids = p.clone();
            ids.extend(c);

            // Use i32 labels to match selective_log_softmax dtype handling
            let mut labels = vec![-100i32; p.len()];
            labels.extend(c.iter().map(|&id| id as i32));

            let pad_len = max_len - ids.len();
            ids.extend(vec![0; pad_len]);
            labels.extend(vec![-100; pad_len]);

            input_ids_vec.extend(ids.iter().map(|&id| id as i32));
            labels_vec.extend(labels);
        }

        let input_ids = Array::from_slice(&input_ids_vec, &[n_completions as i32, max_len as i32]);
        let labels = Array::from_slice(&labels_vec, &[n_completions as i32, max_len as i32]);

        // Temperature for log-prob computation (None = 1.0, no scaling)
        let temperature = if (self.config.temperature - 1.0).abs() > 1e-8 {
            Some(self.config.temperature as f32)
        } else {
            None
        };

        // 1. Compute old_per_token_logps from current policy BEFORE training update.
        //    These are the generation-time log-probs, detached from the gradient graph.
        let old_logits = policy_model
            .forward(&input_ids, None)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let (old_per_token_logps, completion_mask) =
            self.compute_per_token_logps(&old_logits, &labels, temperature)?;
        // Eval to materialize — these must NOT be part of the grad graph
        old_per_token_logps.eval()?;
        completion_mask.eval()?;

        // 2. Compute ref_per_token_logps from reference model (if beta > 0 and ref_model exists)
        let ref_per_token_logps = if self.config.beta > 0.0 {
            if let Some(ref mut ref_m) = ref_model {
                let ref_logits = ref_m.forward(input_ids.clone())?;
                let (ref_logps, _) =
                    self.compute_per_token_logps(&ref_logits, &labels, temperature)?;
                ref_logps.eval()?;
                Some(ref_logps)
            } else {
                None
            }
        } else {
            None
        };

        // 3. Loss function for value_and_grad — only policy model is differentiated
        let loss_fn = |model: &mut M,
                       (input_ids, labels, adv_array, old_logps, mask): (
            &Array,
            &Array,
            &Array,
            &Array,
            &Array,
        )|
         -> std::result::Result<Array, Exception> {
            let logits = model
                .forward(input_ids, None)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let (per_token_logps, _) = self
                .compute_per_token_logps(&logits, labels, temperature)
                .map_err(|e| Exception::custom(e.to_string()))?;

            let (total_loss, _kl, _policy_loss) = self
                .compute_grpo_loss(
                    &per_token_logps,
                    old_logps,
                    ref_per_token_logps.as_ref(),
                    adv_array,
                    mask,
                    None,
                )
                .map_err(|e| Exception::custom(e.to_string()))?;

            Ok(total_loss)
        };

        let (total_loss_arr, grads) = {
            let mut loss_and_grad_fn = nn::value_and_grad(loss_fn);
            loss_and_grad_fn(
                policy_model,
                (
                    &input_ids,
                    &labels,
                    &adv_array,
                    &old_per_token_logps,
                    &completion_mask,
                ),
            )?
        };

        // 4. Update optimizer
        optimizer.update(policy_model, grads)?;

        // Extract loss from the forward pass (already computed, no redundant re-forward)
        let total_loss = total_loss_arr.item::<f32>();

        // Compute KL/policy_loss stats from the values already available in the loss
        // (We use the pre-update values since post-update requires an extra forward pass.
        // The loss value itself is the authoritative training signal.)
        let kl_stat = if ref_per_token_logps.is_some() {
            // Approximate: compute from old policy vs ref (cheap, no extra forward)
            let ref_logps = ref_per_token_logps.as_ref().unwrap();
            let kl_log_ratio = ref_logps.subtract(&old_per_token_logps)?;
            let kl_ratio = kl_log_ratio.exp()?;
            let per_token_kl = kl_ratio
                .subtract(&Array::from_f32(1.0))?
                .subtract(&kl_log_ratio)?;
            let masked_kl = per_token_kl.multiply(&completion_mask)?;
            let safe_count =
                mlx_rs::ops::maximum(&completion_mask.sum(None)?, &Array::from_f32(1.0))?;
            masked_kl.sum(None)?.divide(&safe_count)?.item::<f32>()
        } else {
            0.0
        };

        let mean_reward = if raw_rewards.is_empty() {
            0.0
        } else {
            raw_rewards.iter().sum::<f64>() / raw_rewards.len() as f64
        };
        let mean_adv = if advantages.is_empty() {
            0.0
        } else {
            advantages.iter().sum::<f64>() / advantages.len() as f64
        };

        self.step += 1;

        Ok(GrpoIterationStats {
            step: self.step,
            loss: total_loss,
            kl: kl_stat,
            policy_loss: total_loss, // Policy loss is the dominant component
            reward: mean_reward as f32,
            advantage: mean_adv as f32,
            completions_per_second: n_completions as f32 / start_time.elapsed().as_secs_f32(),
        })
    }

    /// Generate multiple completions for a prompt.
    pub fn generate_completions<M>(
        &mut self,
        model: &mut M,
        prompt_tokens: &[u32],
        tokenizer: &pmetal_data::Tokenizer,
    ) -> GrpoResult<pmetal_models::rl_generation::BatchedGenerationOutput>
    where
        M: TrainableModel,
    {
        let config = BatchedRlConfig {
            num_generations: self.config.num_generations,
            max_new_tokens: self.config.max_completion_length,
            temperature: self.config.temperature as f32,
            top_p: self.config.top_p as f32,
            top_k: self.config.top_k,
            stop_tokens: vec![tokenizer.eos_token_id().unwrap_or(2)],
            seed: None,
            use_prefix_cache: true,
            min_p: 0.05,
        };

        let cache = model
            .create_cache(self.config.max_prompt_length + self.config.max_completion_length)
            .ok_or_else(|| GrpoError::Generation("Model does not support KV cache".into()))?;
        let kv_config = cache.config();

        let mut generator = BatchedRlGenerator::new(config, kv_config.clone());

        generator
            .generate(
                |input, cache| {
                    model
                        .forward_with_cache(input, None, Some(cache))
                        .map_err(|e| Exception::custom(e.to_string()))
                },
                prompt_tokens,
            )
            .map_err(|e| GrpoError::Generation(e.to_string()))
    }

    /// Run full GRPO training loop.
    pub fn run<M, R, O>(
        &mut self,
        policy_model: &mut M,
        mut ref_model: Option<&mut R>,
        tokenizer: &pmetal_data::Tokenizer,
        dataset: &pmetal_data::TrainingDataset,
        reward_fn: &CombinedReward,
        optimizer: &mut O,
    ) -> GrpoResult<()>
    where
        M: TrainableModel,
        R: ModuleParameters + Module<Array, Error = Exception, Output = Array>,
        O: Optimizer,
    {
        info!("Starting GRPO training loop...");
        let n_epochs = self.training_config.num_epochs;

        for epoch in 0..n_epochs {
            info!("Epoch {}/{}", epoch + 1, n_epochs);

            for (i, sample) in dataset.samples().iter().enumerate() {
                let gen_output =
                    self.generate_completions(policy_model, &sample.input_ids, tokenizer)?;

                let prompt_text = tokenizer
                    .decode(&sample.input_ids)
                    .map_err(|e| GrpoError::Tokenizer(e.to_string()))?;
                let mut completions_text = Vec::new();
                for ids in &gen_output.token_ids {
                    let new_ids = &ids[sample.input_ids.len()..];
                    completions_text.push(
                        tokenizer
                            .decode(new_ids)
                            .map_err(|e| GrpoError::Tokenizer(e.to_string()))?,
                    );
                }

                let rewards = reward_fn.compute(
                    &vec![prompt_text; gen_output.token_ids.len()],
                    &completions_text,
                    None,
                )?;

                let mut group =
                    CompletionGroup::new(sample.input_ids.clone(), self.config.num_generations);
                for (j, ids) in gen_output.token_ids.iter().enumerate() {
                    let new_ids = ids[sample.input_ids.len()..].to_vec();
                    group.add_completion(new_ids, rewards[j], gen_output.stopped_by_length[j]);
                }

                let stats = self.train_step(
                    policy_model,
                    ref_model.as_mut().map(|r| &mut **r),
                    &[group],
                    optimizer,
                )?;

                if i % 10 == 0 {
                    info!(
                        "Step {}: loss={:.4}, kl={:.4}, reward={:.4}, completion_len={:.1}",
                        stats.step,
                        stats.loss,
                        stats.kl,
                        stats.reward,
                        gen_output.num_generated.iter().sum::<usize>() as f32
                            / gen_output.num_generated.len() as f32
                    );
                }
            }
        }

        Ok(())
    }
}

/// Trait for GRPO Reward functions.
pub trait RewardFunction: Send + Sync {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>>;
    fn name(&self) -> &str;
}

/// Reward function that checks for proper XML tags (e.g., <thought> and <answer>).
pub struct XmlFormatReward {
    pub tags: Vec<(String, String)>,
}

impl XmlFormatReward {
    pub fn new(tags: Vec<(String, String)>) -> Self {
        Self { tags }
    }

    pub fn default_reasoning() -> Self {
        Self::new(vec![
            ("<thought>".into(), "</thought>".into()),
            ("<answer>".into(), "</answer>".into()),
        ])
    }
}

impl RewardFunction for XmlFormatReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let mut rewards = vec![0.0; completions.len()];
        for (i, completion) in completions.iter().enumerate() {
            let mut score = 0.0;
            for (start_tag, end_tag) in &self.tags {
                if completion.contains(start_tag) && completion.contains(end_tag) {
                    let start_idx = completion.find(start_tag).unwrap();
                    let end_idx = completion.find(end_tag).unwrap();
                    if start_idx < end_idx {
                        score += 0.5;
                    }
                }
            }
            rewards[i] = score;
        }
        Ok(rewards)
    }

    fn name(&self) -> &str {
        "xml_format"
    }
}

/// Reward function that checks for exact matches with ground truth answers.
pub struct AccuracyReward {
    pub answers: Vec<String>,
}

impl AccuracyReward {
    pub fn new(answers: Vec<String>) -> Self {
        Self { answers }
    }
}

impl RewardFunction for AccuracyReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let num_generations = completions.len() / self.answers.len();
        let mut rewards = vec![0.0; completions.len()];

        for (prompt_idx, answer) in self.answers.iter().enumerate() {
            for gen_idx in 0..num_generations {
                let comp_idx = prompt_idx * num_generations + gen_idx;
                let completion = &completions[comp_idx];

                let processed_completion =
                    match (completion.find("<answer>"), completion.find("</answer>")) {
                        (Some(start), Some(end)) if start + 8 <= end => {
                            completion[start + 8..end].trim().to_string()
                        }
                        _ => completion.trim().to_string(),
                    };

                if processed_completion == answer.trim() {
                    rewards[comp_idx] = 1.0;
                }
            }
        }
        Ok(rewards)
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Combined reward function with weights.
pub struct CombinedReward {
    pub functions: Vec<(Box<dyn RewardFunction>, f64)>,
}

impl CombinedReward {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn add(mut self, function: Box<dyn RewardFunction>, weight: f64) -> Self {
        self.functions.push((function, weight));
        self
    }

    pub fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        if self.functions.is_empty() {
            return Err(GrpoError::Reward("No reward functions configured".into()));
        }

        let mut total_rewards = vec![0.0; completions.len()];
        for (func, weight) in &self.functions {
            let rewards = func.compute(prompts, completions, images)?;
            for (i, r) in rewards.iter().enumerate() {
                total_rewards[i] += r * weight;
            }
        }
        Ok(total_rewards)
    }
}

impl Default for CombinedReward {
    fn default() -> Self {
        Self::new()
    }
}
