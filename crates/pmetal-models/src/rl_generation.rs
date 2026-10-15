//! Batched generation for RL training.
//!
//! GRPO samples a group of completions (say 8) from each prompt. Run one
//! after the other, each completion would prefill the prompt and decode on
//! its own, and a decode step at batch size one is bandwidth-bound: the whole
//! model is read to produce a single token. This module does what the group
//! shares once and the rest together:
//!
//! ```text
//! Prompt:  prefill once (from the prefix cache where a repeat allows)
//! Fork:    the prompt's KV state repeated into a batch of `num_generations`
//! Decode:  one [num_generations, 1] forward per step -> one batched sample
//! ```
//!
//! Every completion starts at the same position and advances one token per
//! step, so the batch needs no padding, no per-row offsets and no masks: it
//! is the ordinary cached forward with a batch dimension. A completion that
//! has ended keeps riding along (the step costs the same) and its further
//! tokens are dropped.
//!
//! A completion that ends in a stop token keeps it: a policy gradient needs
//! the stop token in the completion to learn to stop.
use pmetal_runtime::compat::{Array, Exception, ops};
use pmetal_runtime::decode::DecodeSampler;

use pmetal_cache::kv_cache::{KVCache, KVCacheConfig};
use pmetal_cache::prefix_cache::PrefixCache;

use crate::generation::{GenerationConfig, run_cached_prefill_chunks_at};

/// Result type for RL generation.
pub type RlGenResult<T> = Result<T, Exception>;

/// Batched generation output.
#[derive(Debug, Clone)]
pub struct BatchedGenerationOutput {
    /// For each completion, the prompt followed by the generated tokens.
    pub token_ids: Vec<Vec<u32>>,
    /// Number of tokens generated for each sequence.
    pub num_generated: Vec<usize>,
    /// Whether each sequence was stopped by a stop token (which it keeps).
    pub stopped_by_token: Vec<bool>,
    /// Whether each sequence was stopped by max length.
    pub stopped_by_length: Vec<bool>,
}

/// Configuration for batched RL generation.
#[derive(Debug, Clone)]
pub struct BatchedRlConfig {
    /// Number of completions per prompt.
    pub num_generations: usize,
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling; 0 is greedy.
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling (1.0 = disabled).
    pub top_p: f32,
    /// Min-p sampling (0.0 = disabled).
    pub min_p: f32,
    /// Stop token IDs.
    pub stop_tokens: Vec<u32>,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Whether to use prefix caching.
    pub use_prefix_cache: bool,
}

impl Default for BatchedRlConfig {
    fn default() -> Self {
        Self {
            num_generations: 8,
            max_new_tokens: 256,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            stop_tokens: vec![],
            seed: None,
            use_prefix_cache: true,
        }
    }
}

impl BatchedRlConfig {
    /// Create a new config with the given number of generations.
    pub fn new(num_generations: usize) -> Self {
        Self {
            num_generations,
            ..Default::default()
        }
    }

    /// Set maximum new tokens.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set stop tokens.
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<u32>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }

    /// Set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Disable prefix caching.
    pub fn without_prefix_cache(mut self) -> Self {
        self.use_prefix_cache = false;
        self
    }

    /// Convert to GenerationConfig for single-sequence generation.
    pub fn to_generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: 1.0, // Typically disabled for RL
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_tokens: self.stop_tokens.clone(),
            seed: self.seed,
            logit_bias: Vec::new(),
            // Temperature 0 is greedy, not an unscaled draw.
            do_sample: self.temperature > 0.0,
            prefill_step_size: 2048,
        }
    }
}

/// Batched generator for RL training: the completions of a prompt, prefilled
/// once and decoded together. See the module docs.
///
/// Samples through the runtime's [`DecodeSampler`], with the config's
/// temperature, top-k, top-p, min-p and seed, one draw per step for the whole
/// batch.
pub struct BatchedRlGenerator {
    /// Configuration.
    config: BatchedRlConfig,
    /// Prefix cache.
    prefix_cache: Option<PrefixCache>,
    /// KV cache config for creating new caches.
    kv_config: KVCacheConfig,
}

impl BatchedRlGenerator {
    /// Create a new batched RL generator.
    pub fn new(config: BatchedRlConfig, kv_config: KVCacheConfig) -> Self {
        let prefix_cache = config.use_prefix_cache.then(|| PrefixCache::new(32, 0));
        Self {
            config,
            prefix_cache,
            kv_config,
        }
    }

    /// Generate `num_generations` completions of `prompt_tokens`.
    ///
    /// # Arguments
    /// * `forward_fn` - Model forward function: (input_ids `[B, T]`, cache) ->
    ///   logits `[B, T, vocab]`. It runs the prompt once at batch size one,
    ///   then every decode step at batch size `num_generations`.
    /// * `prompt_tokens` - Tokenized prompt
    ///
    /// # Errors
    /// On an empty prompt or no completions, when the model fails, and when
    /// the cache cannot repeat into a batch (TurboQuant).
    pub fn generate<F>(
        &mut self,
        mut forward_fn: F,
        prompt_tokens: &[u32],
    ) -> RlGenResult<BatchedGenerationOutput>
    where
        F: FnMut(&Array, &mut KVCache) -> RlGenResult<Array>,
    {
        let batch = self.config.num_generations;
        let prompt_len = prompt_tokens.len();
        if prompt_len == 0 {
            return Err(Exception::custom("cannot generate from an empty prompt"));
        }
        if batch == 0 {
            return Err(Exception::custom("cannot generate zero completions"));
        }
        let max_new = self.config.max_new_tokens;
        let generation = self.config.to_generation_config();

        // Prefill the prompt once. A prefix-cache hit covers its leading
        // tokens (all but the last when the prompt repeats), so only the
        // rest runs, and what runs also yields the logits of the prompt's
        // last token, which every completion's first token is sampled from.
        let hit = match self.prefix_cache.as_mut() {
            Some(prefix_cache) => {
                prefix_cache.find_longest_prefix(prompt_tokens, &self.kv_config)?
            }
            None => None,
        };
        let (mut prefilled, covered) = match hit {
            Some(hit) => (hit.restored_cache, hit.prefix_len),
            None => (KVCache::new(self.kv_config.clone()), 0),
        };
        let logits = run_cached_prefill_chunks_at(
            &prompt_tokens[covered..],
            covered,
            generation.prefill_step_size,
            |chunk| forward_fn(chunk, &mut prefilled),
        )?;
        // [1, T, V] -> [1, V] at the prompt's last position.
        let last_logits = ops::select_axis(&logits, logits.dim(1) - 1, 1);
        if let Some(prefix_cache) = self.prefix_cache.as_mut() {
            prefix_cache.insert(prompt_tokens, &prefilled);
        }

        // Every completion continues from its own row of the prefilled
        // cache and draws its first token from the same logits.
        let mut cache = prefilled.repeat_batch(batch)?;
        let mut sampler = DecodeSampler::new(generation.sampling_params(), prompt_tokens);
        let vocab = last_logits.dim(-1);
        let mut tokens = sampler.sample(&ops::broadcast_to(&last_logits, &[batch as i32, vocab]));

        let mut sequences: Vec<Vec<u32>> = vec![prompt_tokens.to_vec(); batch];
        let mut stopped_by_token = vec![false; batch];
        let mut finished = vec![max_new == 0; batch];
        for step in 0..max_new {
            // The tokens of this step, on the host: one read for the batch.
            tokens.eval();
            pmetal_runtime::check_unobserved_error()
                .map_err(|e| Exception::custom(format!("rollout sampling: {e}")))?;
            let drawn: Vec<u32> = tokens.as_slice::<u32>().to_vec();
            for (row, &token) in drawn.iter().enumerate() {
                if finished[row] {
                    continue;
                }
                sequences[row].push(token);
                if generation.is_stop_token(token) {
                    stopped_by_token[row] = true;
                    finished[row] = true;
                } else if sequences[row].len() - prompt_len >= max_new {
                    finished[row] = true;
                }
            }
            if finished.iter().all(|&f| f) || step + 1 == max_new {
                break;
            }

            // One forward for the whole batch: row i takes its own token.
            let input = tokens.as_dtype(pmetal_runtime::compat::Dtype::Int32.as_i32());
            let input = input.reshape(&[batch as i32, 1]);
            let logits = forward_fn(&input, &mut cache)?;
            tokens = sampler.sample(&ops::select_axis(&logits, 0, 1));
        }

        let num_generated: Vec<usize> = sequences.iter().map(|s| s.len() - prompt_len).collect();
        let stopped_by_length = stopped_by_token.iter().map(|&stopped| !stopped).collect();
        Ok(BatchedGenerationOutput {
            token_ids: sequences,
            num_generated,
            stopped_by_token,
            stopped_by_length,
        })
    }

    /// Prefix cache statistics: `(hits, misses, hit_rate)`.
    pub fn prefix_cache_stats(&self) -> Option<(usize, usize, f64)> {
        self.prefix_cache
            .as_ref()
            .map(|pc| (pc.hits() as usize, pc.misses() as usize, pc.hit_rate()))
    }

    /// Clear prefix cache.
    pub fn clear_prefix_cache(&mut self) {
        if let Some(ref mut pc) = self.prefix_cache {
            pc.clear();
        }
    }
}

/// Generate multiple completions for RL training.
///
/// This is a convenience function that creates a BatchedRlGenerator and generates
/// completions for a single prompt.
///
/// # Arguments
/// * `forward_fn` - Model forward function
/// * `prompt_tokens` - Tokenized prompt
/// * `config` - Batched RL configuration
/// * `kv_config` - KV cache configuration
pub fn generate_rl_completions<F>(
    forward_fn: F,
    prompt_tokens: &[u32],
    config: BatchedRlConfig,
    kv_config: KVCacheConfig,
) -> RlGenResult<BatchedGenerationOutput>
where
    F: FnMut(&Array, &mut KVCache) -> RlGenResult<Array>,
{
    let mut generator = BatchedRlGenerator::new(config, kv_config);
    generator.generate(forward_fn, prompt_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_kv_config() -> KVCacheConfig {
        KVCacheConfig::new(2, 100, 4, 64)
    }

    #[test]
    fn test_batched_rl_config_default() {
        let config = BatchedRlConfig::default();
        assert_eq!(config.num_generations, 8);
        assert_eq!(config.max_new_tokens, 256);
        assert_eq!(config.temperature, 0.7);
        assert!(config.use_prefix_cache);
    }

    #[test]
    fn test_batched_rl_config_builder() {
        let config = BatchedRlConfig::new(4)
            .with_max_new_tokens(128)
            .with_temperature(0.5)
            .with_stop_tokens(vec![2])
            .with_seed(42);

        assert_eq!(config.num_generations, 4);
        assert_eq!(config.max_new_tokens, 128);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.stop_tokens, vec![2]);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_batched_rl_generator_creation() {
        let config = BatchedRlConfig::default();
        let kv_config = create_test_kv_config();
        let generator = BatchedRlGenerator::new(config, kv_config);

        assert!(generator.prefix_cache.is_some());
    }

    #[test]
    fn test_to_generation_config() {
        let config = BatchedRlConfig::new(8)
            .with_max_new_tokens(100)
            .with_temperature(0.6);

        let gen_config = config.to_generation_config();

        assert_eq!(gen_config.max_new_tokens, 100);
        assert_eq!(gen_config.temperature, 0.6);
        assert!(gen_config.do_sample);
    }

    /// A stand-in model whose logits put 30 on (token + 1) mod 16 at every
    /// position, so its greedy continuation of any prompt counts upward. It
    /// stores a zero key and value per token and row, as a real model stores
    /// its own.
    fn counting_model(input: &Array, cache: &mut KVCache) -> RlGenResult<Array> {
        const VOCAB: usize = 16;
        input.eval();
        let (rows, len) = (input.dim(0) as usize, input.dim(1) as usize);
        let ids: Vec<i32> = input.as_slice::<i32>().to_vec();
        let kv = Array::zeros_f32(&[rows as i32, 4, len as i32, 64]);
        for layer in 0..2 {
            cache.update_and_fetch(layer, &kv, &kv)?;
        }
        let mut logits = vec![0.0f32; ids.len() * VOCAB];
        for (pos, &id) in ids.iter().enumerate() {
            logits[pos * VOCAB + (id as usize + 1) % VOCAB] = 30.0;
        }
        Ok(Array::from_slice(
            &logits,
            &[rows as i32, len as i32, VOCAB as i32],
        ))
    }

    #[test]
    fn rollouts_continue_the_prompt_from_the_model_logits() {
        let config = BatchedRlConfig {
            use_prefix_cache: false,
            temperature: 1.0,
            ..BatchedRlConfig::new(3)
                .with_max_new_tokens(5)
                .with_stop_tokens(vec![15])
                .with_seed(1)
        };
        let mut generator = BatchedRlGenerator::new(config, create_test_kv_config());
        let out = generator.generate(counting_model, &[2, 3, 4]).unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        for (seq, n) in out.token_ids.iter().zip(&out.num_generated) {
            // The first token comes from the last prompt position, every later
            // one from the token before it, and exactly max_new_tokens are made.
            assert_eq!(seq, &vec![2, 3, 4, 5, 6, 7, 8, 9]);
            assert_eq!(*n, 5);
        }
        assert!(out.stopped_by_length.iter().all(|&b| b));

        // A stop token ends the completion early, and stays in it.
        let mut generator = BatchedRlGenerator::new(
            BatchedRlConfig {
                use_prefix_cache: false,
                ..BatchedRlConfig::new(1)
                    .with_max_new_tokens(8)
                    .with_stop_tokens(vec![7])
            },
            create_test_kv_config(),
        );
        let out = generator.generate(counting_model, &[4]).unwrap();
        assert_eq!(out.token_ids[0], vec![4, 5, 6, 7]);
        assert!(out.stopped_by_token[0] && !out.stopped_by_length[0]);
    }

    #[test]
    fn a_prefix_cache_hit_samples_from_the_prompt_logits() {
        // The second rollout of a prompt restores its KV state. It must still
        // produce the first token from the prompt's last logits, not from
        // nothing, and so complete exactly as the first did.
        let config = || BatchedRlConfig {
            temperature: 1.0,
            ..BatchedRlConfig::new(3)
                .with_max_new_tokens(5)
                .with_stop_tokens(vec![15])
                .with_seed(1)
        };
        let mut generator = BatchedRlGenerator::new(config(), create_test_kv_config());
        let cold = generator.generate(counting_model, &[2, 3, 4]).unwrap();
        let warm = generator.generate(counting_model, &[2, 3, 4]).unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        assert_eq!(cold.token_ids[0], vec![2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(warm.token_ids, cold.token_ids);
        assert_eq!(warm.num_generated, cold.num_generated);
        let (_, _, hit_rate) = generator.prefix_cache_stats().unwrap();
        assert!(hit_rate > 0.0, "the second rollout should hit the cache");
    }

    #[test]
    fn the_prompt_is_prefilled_once_for_all_completions() {
        // The completions of one prompt share its KV state: the prompt runs
        // through the model once, however many are sampled, and a repeat of
        // it runs only the token that produces the logits.
        let config = BatchedRlConfig::new(4)
            .with_max_new_tokens(3)
            .with_stop_tokens(vec![15])
            .with_seed(2);
        let mut generator = BatchedRlGenerator::new(config, create_test_kv_config());
        let prompt_runs = std::cell::RefCell::new(Vec::new());
        let run = |input: &Array, cache: &mut KVCache| {
            if input.dim(1) > 1 {
                prompt_runs.borrow_mut().push(input.dim(1));
            }
            counting_model(input, cache)
        };
        generator.generate(&run, &[2, 3, 4, 5]).unwrap();
        assert_eq!(*prompt_runs.borrow(), vec![4], "one prefill of four tokens");
        generator.generate(&run, &[2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            *prompt_runs.borrow(),
            vec![4],
            "the shared four tokens are not run again"
        );
    }

    /// The decode steps are batched: one forward per step for the whole
    /// group, at the group's batch size, instead of one per completion.
    #[test]
    fn the_group_decodes_in_one_forward_per_step() {
        let config = BatchedRlConfig::new(5)
            .with_max_new_tokens(4)
            .with_temperature(0.0)
            .without_prefix_cache();
        let mut generator = BatchedRlGenerator::new(config, create_test_kv_config());
        let shapes = std::cell::RefCell::new(Vec::new());
        let run = |input: &Array, cache: &mut KVCache| {
            shapes.borrow_mut().push((input.dim(0), input.dim(1)));
            counting_model(input, cache)
        };
        let out = generator.generate(&run, &[1, 2, 3]).unwrap();
        // The prompt at batch one, then three steps (the fourth token is
        // sampled from the third step's logits) at batch five.
        assert_eq!(*shapes.borrow(), vec![(1, 3), (5, 1), (5, 1), (5, 1)]);
        assert!(out.token_ids.iter().all(|s| s == &[1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn degenerate_requests_are_errors_or_empty() {
        let mut generator = BatchedRlGenerator::new(
            BatchedRlConfig::new(2).without_prefix_cache(),
            create_test_kv_config(),
        );
        assert!(generator.generate(counting_model, &[]).is_err());
        let mut none = BatchedRlGenerator::new(
            BatchedRlConfig::new(0).without_prefix_cache(),
            create_test_kv_config(),
        );
        assert!(none.generate(counting_model, &[1]).is_err());
        let mut zero = BatchedRlGenerator::new(
            BatchedRlConfig::new(2)
                .with_max_new_tokens(0)
                .without_prefix_cache(),
            create_test_kv_config(),
        );
        let out = zero.generate(counting_model, &[1, 2]).unwrap();
        assert_eq!(out.token_ids, vec![vec![1, 2], vec![1, 2]]);
        assert_eq!(out.num_generated, vec![0, 0]);
    }
}
