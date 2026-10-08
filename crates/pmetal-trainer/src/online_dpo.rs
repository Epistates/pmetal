//! Online DPO (Direct Preference Optimization) trainer.
//!
//! Implements SPPO-style (Self-Play Preference Optimization) iterative training:
//! 1. Generates completions using the current policy
//! 2. Scores completions using a reward model or preference function
//! 3. Creates preference pairs from best/worst completions
//! 4. Updates the policy using DPO loss
//!
//! Based on:
//! - "Direct Preference Optimization" (Rafailov et al., 2023)
//! - "Self-Play Preference Optimization for Language Model Alignment" (Wu et al., 2024)
//!
//! Key insight: Treating alignment as a two-player game and iteratively
//! generating + training converges to Nash equilibrium policy.

use std::sync::Arc;

use pmetal_bridge::compat::{
    Array, Dtype, Exception, indexing::IndexOp, nn, ops, optimizers::Optimizer, random,
};
use pmetal_core::TrainingConfig;
use pmetal_lora::TrainableModel;
use tracing::{debug, info};

use crate::dpo::{DpoConfig, DpoLossType};
use crate::paired_preference::{
    DpoLoss, PairedPreferenceTrainer, PreferenceLoss, ReferenceStrategy,
};

/// Configuration for Online DPO / SPPO.
#[derive(Debug, Clone)]
pub struct OnlineDpoConfig {
    /// Inner DPO configuration.
    pub dpo_config: DpoConfig,

    /// Number of completions to generate per prompt.
    pub num_samples_per_prompt: usize,

    /// Maximum tokens to generate per completion.
    pub max_new_tokens: usize,

    /// Temperature for sampling during generation.
    pub temperature: f32,

    /// Top-p (nucleus) sampling parameter.
    pub top_p: f32,

    /// Number of online iterations (generate -> train cycles).
    pub num_iterations: usize,

    /// Number of gradient steps per iteration.
    pub steps_per_iteration: usize,

    /// Whether to use the previous policy checkpoint as reference.
    /// If true, implements SPPO-style self-play.
    pub use_self_play: bool,

    /// How often to update the reference model (in iterations).
    /// Only used when use_self_play is true.
    pub ref_update_interval: usize,

    /// Minimum reward margin to create a preference pair.
    /// Pairs with smaller margin are discarded.
    pub min_reward_margin: f32,
}

impl Default for OnlineDpoConfig {
    fn default() -> Self {
        Self {
            dpo_config: DpoConfig::default(),
            num_samples_per_prompt: 4,
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            num_iterations: 10,
            steps_per_iteration: 100,
            use_self_play: true,
            ref_update_interval: 1,
            min_reward_margin: 0.1,
        }
    }
}

impl OnlineDpoConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the DPO beta parameter.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.dpo_config.beta = beta;
        self
    }

    /// Enable SimPO mode (reference-free with margin).
    pub fn with_simpo(mut self, gamma: f64) -> Self {
        self.dpo_config.loss_type = DpoLossType::SimPo;
        self.dpo_config.simpo_gamma = gamma;
        self.dpo_config.reference_free = true;
        self
    }
}

/// Trait for scoring completions.
pub trait RewardFunction: Send + Sync {
    /// Score a batch of completions.
    ///
    /// # Arguments
    /// * `prompt_tokens` - The prompt tokens [batch, prompt_len]
    /// * `completion_tokens` - The completion tokens [batch, completion_len]
    ///
    /// # Returns
    /// Array of scalar scores [batch]
    fn score(&self, prompt_tokens: &Array, completion_tokens: &Array) -> Result<Array, Exception>;
}

/// A preference pair for training.
#[derive(Debug, Clone)]
pub struct OnlinePreferencePair {
    /// Prompt token IDs.
    pub prompt_ids: Vec<u32>,
    /// Chosen (winner) completion token IDs.
    pub chosen_ids: Vec<u32>,
    /// Rejected (loser) completion token IDs.
    pub rejected_ids: Vec<u32>,
    /// Reward for chosen completion.
    pub chosen_reward: f32,
    /// Reward for rejected completion.
    pub rejected_reward: f32,
}

/// Statistics for an online DPO iteration.
#[derive(Debug, Clone, Default)]
pub struct OnlineDpoIterationStats {
    /// Iteration number.
    pub iteration: usize,
    /// Average loss over the iteration.
    pub avg_loss: f32,
    /// Average reward margin (chosen - rejected).
    pub avg_reward_margin: f32,
    /// Number of preference pairs generated.
    pub num_pairs: usize,
    /// Average chosen reward.
    pub avg_chosen_reward: f32,
    /// Average rejected reward.
    pub avg_rejected_reward: f32,
    /// Accuracy (fraction where chosen > rejected after training).
    pub accuracy: f32,
}

/// Online DPO / SPPO Trainer.
///
/// Implements iterative preference optimization where:
/// 1. The current policy generates multiple completions per prompt
/// 2. Completions are scored by a reward function
/// 3. Best/worst pairs are selected for DPO training
/// 4. The policy is updated on these pairs
///
/// In SPPO mode (use_self_play=true), the reference model is periodically
/// updated to the current policy, creating a self-play dynamic.
pub struct OnlineDpoTrainer {
    config: OnlineDpoConfig,
    #[allow(dead_code)] // Needed for future distributed/checkpoint support
    training_config: TrainingConfig,
    reward_func: Arc<dyn RewardFunction>,
    step: usize,
    iteration: usize,
}

#[allow(dead_code)] // Experimental trainer — methods used internally once train() is wired
impl OnlineDpoTrainer {
    /// Create a new Online DPO trainer.
    pub fn new(
        config: OnlineDpoConfig,
        training_config: TrainingConfig,
        reward_func: Arc<dyn RewardFunction>,
    ) -> Self {
        Self {
            config,
            training_config,
            reward_func,
            step: 0,
            iteration: 0,
        }
    }

    /// Generate completions for a batch of prompts.
    ///
    /// Returns: Vec of (prompt_tokens, Vec<completion_tokens>)
    pub fn generate_completions<M: TrainableModel>(
        &self,
        model: &mut M,
        prompts: &[Vec<u32>],
    ) -> Result<Vec<(Vec<u32>, Vec<Vec<u32>>)>, Exception> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let mut completions = Vec::with_capacity(self.config.num_samples_per_prompt);

            for _ in 0..self.config.num_samples_per_prompt {
                // Create input array
                let input_ids = Array::from_slice(
                    &prompt.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                    &[1, prompt.len() as i32],
                );

                // Generate tokens autoregressively
                let mut generated = prompt.clone();
                let mut current_ids = input_ids.clone();

                for _ in 0..self.config.max_new_tokens {
                    // Forward pass
                    let logits = model
                        .forward(&current_ids, None)
                        .map_err(|e| Exception::custom(e.to_string()))?;

                    // Get last token logits
                    let last_logits = logits.index((.., -1, ..));

                    // Apply temperature scaling to logits
                    let scaled =
                        last_logits.multiply(&Array::from_f32(1.0 / self.config.temperature));

                    // Sample from categorical distribution (proper stochastic sampling)
                    // categorical() takes logits directly (applies softmax internally)
                    let mut next_token = scaled.categorical();
                    next_token.eval();
                    let token_id = next_token.clone().item_u32();

                    generated.push(token_id);

                    // Check for EOS (assuming 2 is EOS, should be configurable)
                    if token_id == 2 {
                        break;
                    }

                    // Update input for next iteration
                    current_ids = Array::from_slice(
                        &generated.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                        &[1, generated.len() as i32],
                    );
                }

                // Extract just the completion (without prompt)
                let completion: Vec<u32> = generated[prompt.len()..].to_vec();
                completions.push(completion);
            }

            results.push((prompt.clone(), completions));
        }

        Ok(results)
    }

    /// Score completions and create preference pairs.
    pub fn create_preference_pairs(
        &self,
        prompt_completions: &[(Vec<u32>, Vec<Vec<u32>>)],
    ) -> Result<Vec<OnlinePreferencePair>, Exception> {
        let mut pairs = Vec::new();

        for (prompt, completions) in prompt_completions {
            if completions.len() < 2 {
                continue;
            }

            // Score each completion
            let prompt_array = Array::from_slice(
                &prompt.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                &[1, prompt.len() as i32],
            );

            let mut scored: Vec<(usize, f32)> = Vec::with_capacity(completions.len());

            for (idx, completion) in completions.iter().enumerate() {
                let completion_array = Array::from_slice(
                    &completion.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                    &[1, completion.len() as i32],
                );

                let mut score = self.reward_func.score(&prompt_array, &completion_array)?;
                score.eval();
                scored.push((idx, score.item_f32()));
            }

            // Sort by score (descending)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Create pairs: best vs each worse completion
            let best_idx = scored[0].0;
            let best_reward = scored[0].1;

            for &(idx, reward) in scored.iter().skip(1) {
                let margin = best_reward - reward;

                if margin >= self.config.min_reward_margin {
                    pairs.push(OnlinePreferencePair {
                        prompt_ids: prompt.clone(),
                        chosen_ids: completions[best_idx].clone(),
                        rejected_ids: completions[idx].clone(),
                        chosen_reward: best_reward,
                        rejected_reward: reward,
                    });
                }
            }
        }

        Ok(pairs)
    }

    /// Run a single training step on a preference pair.
    pub fn train_step<M, O>(
        &mut self,
        policy_model: &mut M,
        ref_model: &mut M,
        pair: &OnlinePreferencePair,
        optimizer: &mut O,
    ) -> Result<f32, Exception>
    where
        M: TrainableModel,
        O: Optimizer,
    {
        // Build input sequences
        let prompt_len = pair.prompt_ids.len();

        // Chosen: prompt + chosen_completion
        let mut chosen_full: Vec<i32> = pair.prompt_ids.iter().map(|&x| x as i32).collect();
        chosen_full.extend(pair.chosen_ids.iter().map(|&x| x as i32));

        // Rejected: prompt + rejected_completion
        let mut rejected_full: Vec<i32> = pair.prompt_ids.iter().map(|&x| x as i32).collect();
        rejected_full.extend(pair.rejected_ids.iter().map(|&x| x as i32));

        // Labels: -100 for prompt, actual IDs for completion
        let mut chosen_labels: Vec<i64> = vec![-100i64; prompt_len];
        chosen_labels.extend(pair.chosen_ids.iter().map(|&x| x as i64));

        let mut rejected_labels: Vec<i64> = vec![-100i64; prompt_len];
        rejected_labels.extend(pair.rejected_ids.iter().map(|&x| x as i64));

        // Create arrays
        let chosen_ids = Array::from_slice(&chosen_full, &[1, chosen_full.len() as i32]);
        let rejected_ids = Array::from_slice(&rejected_full, &[1, rejected_full.len() as i32]);
        let chosen_labels_arr = Array::from_slice(&chosen_labels, &[1, chosen_labels.len() as i32]);
        let rejected_labels_arr =
            Array::from_slice(&rejected_labels, &[1, rejected_labels.len() as i32]);

        // Delegate the grad/optimizer step to the shared
        // PairedPreferenceTrainer<DpoLoss>. Reference log-probs come from
        // an external ref_model forward only when the DPO config actually
        // calls for them (Precomputed strategy); stop-gradient and
        // reference-free paths skip the ref forward entirely.
        let loss_kernel = DpoLoss::new(self.config.dpo_config.clone());
        let precomputed_ref = match loss_kernel.reference_strategy() {
            ReferenceStrategy::Precomputed => {
                let ref_chosen_logits = ref_model
                    .forward(&chosen_ids, None)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let ref_rejected_logits = ref_model
                    .forward(&rejected_ids, None)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                let ref_chosen_logps =
                    loss_kernel.reduce_log_probs(&ref_chosen_logits, &chosen_labels_arr)?;
                let ref_rejected_logps =
                    loss_kernel.reduce_log_probs(&ref_rejected_logits, &rejected_labels_arr)?;
                Some((ref_chosen_logps, ref_rejected_logps))
            }
            _ => None,
        };

        let inner = PairedPreferenceTrainer::new(loss_kernel, TrainingConfig::default());
        let (loss_val, _metrics) = inner.step(
            policy_model,
            optimizer,
            &chosen_ids,
            &chosen_labels_arr,
            &rejected_ids,
            &rejected_labels_arr,
            precomputed_ref,
        )?;

        self.step += 1;
        Ok(loss_val)
    }

    /// Run the full online training loop.
    pub fn train<M, O>(
        &mut self,
        policy_model: &mut M,
        ref_model: &mut M,
        prompts: &[Vec<u32>],
        optimizer: &mut O,
    ) -> Result<Vec<OnlineDpoIterationStats>, Exception>
    where
        M: TrainableModel + Clone,
        O: Optimizer,
    {
        let mut all_stats = Vec::with_capacity(self.config.num_iterations);

        info!(
            "Starting Online DPO training for {} iterations",
            self.config.num_iterations
        );

        for iter in 0..self.config.num_iterations {
            self.iteration = iter;
            info!(
                "=== Iteration {}/{} ===",
                iter + 1,
                self.config.num_iterations
            );

            // 1. Generate completions
            debug!("Generating completions...");
            let prompt_completions = self.generate_completions(policy_model, prompts)?;
            debug!(
                "Generated {} prompt-completion sets",
                prompt_completions.len()
            );

            // 2. Create preference pairs
            debug!("Creating preference pairs...");
            let pairs = self.create_preference_pairs(&prompt_completions)?;
            info!("Created {} preference pairs", pairs.len());

            if pairs.is_empty() {
                info!("No valid pairs, skipping iteration");
                continue;
            }

            // 3. Train on pairs
            let mut total_loss = 0.0;
            let mut total_chosen_reward = 0.0;
            let mut total_rejected_reward = 0.0;

            let steps = self.config.steps_per_iteration.min(pairs.len());

            for step in 0..steps {
                let pair_idx = step % pairs.len();
                let pair = &pairs[pair_idx];

                let loss = self.train_step(policy_model, ref_model, pair, optimizer)?;
                total_loss += loss;
                total_chosen_reward += pair.chosen_reward;
                total_rejected_reward += pair.rejected_reward;

                if (step + 1) % 10 == 0 {
                    debug!("  Step {}/{}: loss={:.4}", step + 1, steps, loss);
                }
            }

            // 4. Update reference model if using self-play
            if self.config.use_self_play && (iter + 1) % self.config.ref_update_interval == 0 {
                info!("Updating reference model to current policy");
                *ref_model = policy_model.clone();
            }

            // Compute stats
            let avg_loss = total_loss / steps as f32;
            let avg_chosen_reward = total_chosen_reward / steps as f32;
            let avg_rejected_reward = total_rejected_reward / steps as f32;
            let avg_reward_margin = avg_chosen_reward - avg_rejected_reward;

            let stats = OnlineDpoIterationStats {
                iteration: iter,
                avg_loss,
                avg_reward_margin,
                num_pairs: pairs.len(),
                avg_chosen_reward,
                avg_rejected_reward,
                accuracy: 0.0, // Would need evaluation to compute
            };

            info!(
                "Iteration {} complete: loss={:.4}, margin={:.4}, pairs={}",
                iter + 1,
                avg_loss,
                avg_reward_margin,
                pairs.len()
            );

            all_stats.push(stats);
        }

        info!("Online DPO training complete!");
        Ok(all_stats)
    }

    /// Get current step count.
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Get current iteration.
    pub fn current_iteration(&self) -> usize {
        self.iteration
    }
}

/// Simple reward function that uses length as a proxy (for testing).
pub struct LengthRewardFunction {
    /// Target length for completions.
    pub target_length: usize,
    /// How much to penalize deviation from target.
    pub penalty_scale: f32,
}

impl Default for LengthRewardFunction {
    fn default() -> Self {
        Self {
            target_length: 100,
            penalty_scale: 0.01,
        }
    }
}

impl RewardFunction for LengthRewardFunction {
    fn score(&self, _prompt_tokens: &Array, completion_tokens: &Array) -> Result<Array, Exception> {
        let len = completion_tokens.dim(1) as f32;
        let target = self.target_length as f32;
        let deviation = (len - target).abs();
        let reward = 1.0 - deviation * self.penalty_scale;
        Ok(Array::from_f32(reward.max(0.0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_dpo_config() {
        let config = OnlineDpoConfig::new().with_beta(0.2).with_simpo(1.5);

        assert_eq!(config.dpo_config.beta, 0.2);
        assert!(matches!(config.dpo_config.loss_type, DpoLossType::SimPo));
        assert_eq!(config.dpo_config.simpo_gamma, 1.5);
        assert!(config.dpo_config.reference_free);
    }

    #[test]
    fn test_length_reward_function() {
        let reward_fn = LengthRewardFunction {
            target_length: 50,
            penalty_scale: 0.02,
        };

        // Perfect length
        let prompt = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        let completion = Array::from_slice(&vec![1_i32; 50], &[1, 50]);

        let score = reward_fn.score(&prompt, &completion).unwrap();
        score.eval().unwrap();
        assert!((score.item_f32() - 1.0).abs() < 0.01);

        // Too short (length 10, deviation 40)
        let short = Array::from_slice(&vec![1_i32; 10], &[1, 10]);
        let score_short = reward_fn.score(&prompt, &short).unwrap();
        score_short.eval().unwrap();
        // reward = 1.0 - 40 * 0.02 = 0.2
        assert!((score_short.item_f32() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_preference_pair_creation() {
        // This test just verifies the structure compiles
        // Full integration test would need a real model
        let config = OnlineDpoConfig::default();
        let reward_fn = Arc::new(LengthRewardFunction::default());
        let training_config = TrainingConfig::default();

        let _trainer = OnlineDpoTrainer::new(config, training_config, reward_fn);
    }
}
