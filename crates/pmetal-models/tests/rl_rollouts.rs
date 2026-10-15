//! A group of rollouts decoded as one batch is the group decoded one
//! completion at a time.
//!
//! The model is a small randomly initialized Qwen3 in f32: its weights mean
//! nothing, which is the point, because every token it picks is decided by
//! the whole forward pass, the KV cache and the rope offsets. Greedy rollouts
//! must match plain greedy generation token for token, whatever the batch
//! size; sampled ones must come out as independent draws from the same
//! distribution; and the prompt must run through the model once.

use pmetal_cache::kv_cache::{KVCache, KVCacheConfig};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use pmetal_models::generation::{GenerationConfig, generate_cached};
use pmetal_models::rl_generation::{BatchedRlConfig, BatchedRlGenerator};
use pmetal_runtime::compat::{Array, Exception};
use serial_test::serial;

const PROMPT: [u32; 5] = [7, 19, 3, 41, 12];

fn tiny_model() -> Qwen3ForCausalLM {
    let config = Qwen3Config {
        hidden_size: 48,
        intermediate_size: 96,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 16,
        vocab_size: 64,
        ..Default::default()
    };
    Qwen3ForCausalLM::new(config).expect("tiny qwen3")
}

fn kv_config() -> KVCacheConfig {
    KVCacheConfig::new(3, 128, 2, 16)
}

fn forward(
    model: &mut Qwen3ForCausalLM,
    input: &Array,
    cache: &mut KVCache,
) -> Result<Array, Exception> {
    model.forward_with_cache(input, None, Some(cache))
}

#[test]
#[serial]
fn greedy_rollouts_are_greedy_generation_whatever_the_batch_size() {
    let mut model = tiny_model();
    let reference = generate_cached(
        |input, cache| forward(&mut model, input, cache),
        &PROMPT,
        GenerationConfig::greedy(20),
        &mut KVCache::new(kv_config()),
    )
    .unwrap();
    assert_eq!(reference.num_generated, 20);
    // The tiny model must have something to say, or the test proves nothing.
    let made = &reference.token_ids[PROMPT.len()..];
    assert!(made.iter().any(|&t| t != made[0]), "{made:?}");

    for batch in [1usize, 2, 5] {
        let config = BatchedRlConfig::new(batch)
            .with_max_new_tokens(20)
            .with_temperature(0.0)
            .without_prefix_cache();
        let out = BatchedRlGenerator::new(config, kv_config())
            .generate(|input, cache| forward(&mut model, input, cache), &PROMPT)
            .unwrap();
        pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
        assert_eq!(out.token_ids.len(), batch);
        for (row, tokens) in out.token_ids.iter().enumerate() {
            assert_eq!(tokens, &reference.token_ids, "batch {batch}, row {row}");
        }
    }
}

#[test]
#[serial]
fn a_stop_token_ends_one_completion_and_not_the_others() {
    let mut model = tiny_model();
    let greedy = generate_cached(
        |input, cache| forward(&mut model, input, cache),
        &PROMPT,
        GenerationConfig::greedy(12),
        &mut KVCache::new(kv_config()),
    )
    .unwrap();
    let made = greedy.token_ids[PROMPT.len()..].to_vec();
    let stop = made[5];
    let at = made.iter().position(|&t| t == stop).unwrap();

    // Sampled at a high temperature the rows diverge, so some of them hit
    // the stop token and some don't; every row that did ends on it, keeps
    // it, and every one that didn't ran the full length.
    let config = BatchedRlConfig {
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        ..BatchedRlConfig::new(16)
            .with_max_new_tokens(12)
            .with_temperature(1.5)
            .with_stop_tokens(vec![stop])
            .with_seed(3)
            .without_prefix_cache()
    };
    let out = BatchedRlGenerator::new(config, kv_config())
        .generate(|input, cache| forward(&mut model, input, cache), &PROMPT)
        .unwrap();
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    let _ = at;
    let mut stopped = 0;
    for row in 0..16 {
        let tokens = &out.token_ids[row][PROMPT.len()..];
        assert_eq!(tokens.len(), out.num_generated[row]);
        if out.stopped_by_token[row] {
            stopped += 1;
            assert_eq!(
                tokens.last(),
                Some(&stop),
                "row {row} ends on the stop token"
            );
            assert_eq!(
                tokens.iter().filter(|&&t| t == stop).count(),
                1,
                "row {row}"
            );
            assert!(!out.stopped_by_length[row]);
        } else {
            assert_eq!(tokens.len(), 12, "row {row} ran to the limit");
            assert!(!tokens.contains(&stop), "row {row}");
            assert!(out.stopped_by_length[row]);
        }
    }
    assert!(stopped > 0 && stopped < 16, "{stopped} of 16 rows stopped");
    let distinct: std::collections::HashSet<_> = out.token_ids.iter().collect();
    assert!(distinct.len() > 8, "sampled rows are independent draws");
}

#[test]
#[serial]
fn the_prompt_runs_through_the_model_once_for_the_whole_group() {
    let mut model = tiny_model();
    let config = BatchedRlConfig::new(6)
        .with_max_new_tokens(5)
        .with_temperature(0.0);
    let mut generator = BatchedRlGenerator::new(config, kv_config());
    let shapes = std::cell::RefCell::new(Vec::new());
    let mut run = |input: &Array, cache: &mut KVCache| {
        shapes.borrow_mut().push((input.dim(0), input.dim(1)));
        forward(&mut model, input, cache)
    };
    let first = generator.generate(&mut run, &PROMPT).unwrap();
    // The prompt once at batch one, then four batched steps.
    assert_eq!(
        *shapes.borrow(),
        vec![(1, 5), (6, 1), (6, 1), (6, 1), (6, 1)]
    );
    // A repeat of the prompt restores it: only its last token runs again.
    shapes.borrow_mut().clear();
    let second = generator.generate(&mut run, &PROMPT).unwrap();
    assert_eq!(
        *shapes.borrow(),
        vec![(1, 1), (6, 1), (6, 1), (6, 1), (6, 1)]
    );
    assert_eq!(second.token_ids, first.token_ids);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
}

#[test]
#[serial]
fn a_quantized_rollout_cache_decodes_the_group() {
    // Quantized KV groups start at 32 channels, so this model has a head
    // dimension of 32.
    let mut model = Qwen3ForCausalLM::new(Qwen3Config {
        hidden_size: 64,
        intermediate_size: 96,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: Some(1),
        head_dim: 32,
        vocab_size: 64,
        ..Default::default()
    })
    .expect("tiny qwen3");
    let cache = || KVCacheConfig::new(2, 128, 1, 32).with_quantized(8, 32);
    let rollout = |model: &mut Qwen3ForCausalLM, batch: usize| {
        let config = BatchedRlConfig::new(batch)
            .with_max_new_tokens(8)
            .with_temperature(0.0)
            .without_prefix_cache();
        BatchedRlGenerator::new(config, cache())
            .generate(|input, cache| forward(model, input, cache), &PROMPT)
            .unwrap()
    };
    let single = rollout(&mut model, 1);
    let group = rollout(&mut model, 3);
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");
    assert_eq!(group.num_generated, vec![8, 8, 8]);
    for tokens in &group.token_ids {
        assert_eq!(tokens, &single.token_ids[0]);
    }
}
