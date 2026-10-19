//! Sampling completions from a policy being trained: the one rollout RL and
//! on-policy distillation share.
//!
//! The completions of a prompt are decoded together as one batch with the
//! prompt prefilled once (see
//! [`BatchedRlGenerator`](pmetal_models::rl_generation::BatchedRlGenerator)),
//! on the model's KV cache, through the runtime's one decode sampler.

use pmetal_lora::TrainableModel;
use pmetal_models::rl_generation::{BatchedGenerationOutput, BatchedRlConfig, BatchedRlGenerator};
use pmetal_runtime::compat::Exception;

/// How to sample completions.
#[derive(Debug, Clone)]
pub struct RolloutConfig {
    /// Completions per prompt, decoded as one batch.
    pub num_generations: usize,
    /// Longest completion, in tokens.
    pub max_new_tokens: usize,
    /// Sampling temperature (greater than 0).
    pub temperature: f32,
    /// Top-p; 1 disables it.
    pub top_p: f32,
    /// Top-k; 0 disables it.
    pub top_k: usize,
    /// Min-p; 0 disables it.
    pub min_p: f32,
    /// Tokens that end a completion (and stay in it).
    pub stop_tokens: Vec<u32>,
    /// Seed of this rollout's random stream.
    pub seed: Option<u64>,
    /// The KV cache's length: the prompt and the completion.
    pub max_len: usize,
    /// Quantize the KV cache to this many bits (2, 4 or 8).
    pub kv_cache_bits: Option<u8>,
}

/// Sample `config.num_generations` completions of `prompt` from `model`.
pub fn sample_completions<M: TrainableModel>(
    model: &mut M,
    prompt: &[u32],
    config: &RolloutConfig,
) -> Result<BatchedGenerationOutput, Exception> {
    let rl_config = BatchedRlConfig {
        num_generations: config.num_generations,
        max_new_tokens: config.max_new_tokens,
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        stop_tokens: config.stop_tokens.clone(),
        seed: config.seed,
        use_prefix_cache: true,
        min_p: config.min_p,
    };

    let cache = if let Some(bits) = config.kv_cache_bits {
        let mode = pmetal_cache::kv_cache::CacheMode::Quantized {
            bits,
            group_size: 64,
        };
        model.create_cache_with_mode(config.max_len, mode)
    } else {
        model.create_cache(config.max_len)
    }
    .ok_or_else(|| Exception::custom("Model does not support KV cache"))?;
    let kv_config = cache.config();

    let mut generator = BatchedRlGenerator::new(rl_config, kv_config.clone());
    generator.generate(
        |input, cache| {
            model
                .forward_with_cache_last(input, None, Some(cache))
                .map_err(|e| Exception::custom(e.to_string()))
        },
        prompt,
    )
}
