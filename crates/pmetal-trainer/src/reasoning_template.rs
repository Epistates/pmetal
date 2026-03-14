//! Pre-configured templates for Agentic RL and Reasoning models.
//!
//! Provides "Reasoning-Model-in-a-Box" configurations for common tasks like
//! mathematics and coding, using GRPO and RBKD.

use crate::grpo::{
    AccuracyReward, CombinedReward, GrpoConfig, GrpoLossType, RewardFunction, XmlFormatReward,
};

/// Template for training a mathematical reasoning model.
pub struct MathReasoningTemplate;

impl MathReasoningTemplate {
    /// Returns a standard GRPO configuration for math reasoning.
    pub fn config() -> GrpoConfig {
        GrpoConfig {
            num_generations: 8,
            max_completion_length: 1024,
            max_prompt_length: 512,
            beta: 0.05,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            whiten_advantages: true,
            entropy_coef: 0.001,
            loss_type: GrpoLossType::Bnpo,
            epsilon_low: 0.2,
            epsilon_high: 0.2,
        }
    }

    /// Returns a combined reward function for math reasoning.
    ///
    /// Combines:
    /// - XML format reward (ensures <thought> and <answer> tags)
    /// - Accuracy reward (exact match or \boxed{} comparison)
    pub fn reward(answers: Vec<String>) -> CombinedReward {
        CombinedReward::new()
            .add(Box::new(XmlFormatReward::default_reasoning()), 0.2)
            .add(Box::new(AccuracyReward::new(answers)), 0.8)
    }
}

/// Template for training a coding reasoning model.
pub struct CodeReasoningTemplate;

impl CodeReasoningTemplate {
    /// Returns a standard GRPO configuration for code reasoning.
    pub fn config() -> GrpoConfig {
        GrpoConfig {
            num_generations: 4, // Code generation is more expensive
            max_completion_length: 2048,
            max_prompt_length: 1024,
            beta: 0.1,
            temperature: 0.6,
            top_p: 0.9,
            top_k: 40,
            whiten_advantages: true,
            entropy_coef: 0.0,
            loss_type: GrpoLossType::DrGrpo, // Use detailed reward for code
            epsilon_low: 0.2,
            epsilon_high: 0.2,
        }
    }

    /// Returns a combined reward function for code reasoning.
    pub fn reward(test_cases: Vec<String>) -> CombinedReward {
        CombinedReward::new()
            .add(Box::new(XmlFormatReward::default_reasoning()), 0.1)
            .add(Box::new(CodeExecutionReward::new(test_cases)), 0.9)
    }
}

/// Reward function that simulates code execution (Stub for Q1 2026).
/// In production, this would interface with a secure sandbox.
pub struct CodeExecutionReward {
    pub test_cases: Vec<String>,
}

impl CodeExecutionReward {
    pub fn new(test_cases: Vec<String>) -> Self {
        Self { test_cases }
    }
}

impl RewardFunction for CodeExecutionReward {
    fn compute(
        &self,
        _prompts: &[String],
        completions: &[String],
        _images: Option<&[Vec<mlx_rs::Array>]>,
    ) -> crate::grpo::GrpoResult<Vec<f64>> {
        let mut rewards = vec![0.0; completions.len()];
        for (i, completion) in completions.iter().enumerate() {
            // Structural reward: check for fenced code blocks (opening + closing)
            let fence_count = completion.matches("```").count();
            if fence_count >= 2 {
                rewards[i] += 0.25; // Has code block structure
            }
            // Check if any test case patterns appear in the output
            for test in &self.test_cases {
                if completion.contains(test) {
                    rewards[i] += 0.25 / self.test_cases.len().max(1) as f64;
                }
            }
        }
        Ok(rewards)
    }

    fn name(&self) -> &str {
        "code_execution"
    }
}
