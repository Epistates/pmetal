//! The native engine applies the repetition, frequency and presence
//! penalties as the dynamic model path does: greedy decoding under them
//! picks the same tokens on both.
//!
//! The native decode loop used to apply only the frequency and presence
//! penalties, one token late and never counting the first token, and the
//! repetition penalty not at all, so `--repetition-penalty` did nothing on
//! the engine `pmetal infer` runs for Qwen3, Qwen3.5, Llama 4, DeepSeek,
//! gpt-oss and Gemma 4. The repetition penalty covers the prompt as well as
//! the output, as transformers' `RepetitionPenaltyLogitsProcessor` does.
//!
//! The model is the tiny Qwen3.5 checkpoint the parity suite ships. Its
//! greedy decode repeats itself, so every penalty changes the tokens.

use std::path::PathBuf;

use pmetal_bridge::decode::SamplingParams;
use pmetal_models::DynamicModel;
use pmetal_models::generation::{
    GenerationConfig, generate_cached, generate_cached_async_streaming,
};

const PROMPT: [u32; 6] = [3, 11, 7, 2, 9, 40];
const NEW_TOKENS: usize = 24;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("qwen3_5_swish_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path("qwen3_5_swish_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    dir
}

#[derive(Clone, Copy, Debug)]
struct Penalties {
    repetition: f32,
    frequency: f32,
    presence: f32,
}

const NONE: Penalties = Penalties {
    repetition: 1.0,
    frequency: 0.0,
    presence: 0.0,
};

fn native(penalties: Penalties) -> Vec<u32> {
    use pmetal_bridge::qwen3_native::{
        NativeCache, generate, load_config, load_model, prefill_first_token,
    };
    let dir = checkpoint();
    let config = load_config(dir.path()).expect("config");
    let weights = load_model(dir.path(), &config).expect("weights");
    let params = SamplingParams {
        repetition_penalty: penalties.repetition,
        frequency_penalty: penalties.frequency,
        presence_penalty: penalties.presence,
        ..SamplingParams::new(0.0)
    };
    let mut cache = NativeCache::new_empty(&weights);
    let (first, sampler) = prefill_first_token(&weights, &mut cache, &PROMPT, params);
    let (rest, _) = generate(&weights, &mut cache, first, NEW_TOKENS - 1, sampler, |_| {
        true
    });
    pmetal_bridge::check_unobserved_error().expect("native decode: no bridge op failed");
    std::iter::once(first).chain(rest).collect()
}

fn greedy_config(penalties: Penalties) -> GenerationConfig {
    let mut config = GenerationConfig::greedy(NEW_TOKENS);
    config.repetition_penalty = penalties.repetition;
    config.frequency_penalty = penalties.frequency;
    config.presence_penalty = penalties.presence;
    config
}

/// The dynamic model, sampling on the host one token at a time.
fn dynamic(penalties: Penalties) -> Vec<u32> {
    let dir = checkpoint();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut cache = model.create_cache(PROMPT.len() + NEW_TOKENS + 8);
    let mut mamba = model.create_mamba_cache();
    let out = generate_cached(
        |input, cache| model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut()),
        &PROMPT,
        greedy_config(penalties),
        &mut cache,
    )
    .expect("generate");
    pmetal_bridge::check_unobserved_error().expect("dynamic decode: no bridge op failed");
    out.token_ids[PROMPT.len()..].to_vec()
}

/// The dynamic model through the pipelined loop `pmetal infer` runs it on.
fn dynamic_pipelined(penalties: Penalties) -> Vec<u32> {
    let dir = checkpoint();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut cache = model.create_cache(PROMPT.len() + NEW_TOKENS + 8);
    let mut mamba = model.create_mamba_cache();
    let out = generate_cached_async_streaming(
        |input, cache| model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut()),
        &PROMPT,
        greedy_config(penalties),
        &mut cache,
        |_| true,
    )
    .expect("generate");
    pmetal_bridge::check_unobserved_error().expect("pipelined decode: no bridge op failed");
    out.token_ids[PROMPT.len()..].to_vec()
}

fn check(penalties: Penalties) {
    let want = dynamic(penalties);
    assert_eq!(want.len(), NEW_TOKENS);
    assert_ne!(
        want,
        dynamic(NONE),
        "{penalties:?} leaves the greedy tokens as they were, so it checks nothing"
    );
    assert_eq!(native(penalties), want, "native engine, {penalties:?}");
    assert_eq!(
        dynamic_pipelined(penalties),
        want,
        "pipelined dynamic model, {penalties:?}"
    );
}

#[test]
fn unpenalized_greedy_decoding_agrees() {
    let want = dynamic(NONE);
    assert_eq!(native(NONE), want);
    assert_eq!(dynamic_pipelined(NONE), want);
}

#[test]
fn the_repetition_penalty_is_applied_on_every_engine() {
    check(Penalties {
        repetition: 1.3,
        ..NONE
    });
}

/// Rewarding repeats (a penalty below 1) is the same rule run backwards.
#[test]
fn a_repetition_reward_is_applied_on_every_engine() {
    check(Penalties {
        repetition: 0.7,
        ..NONE
    });
}

#[test]
fn frequency_and_presence_penalties_are_applied_on_every_engine() {
    check(Penalties {
        repetition: 1.1,
        frequency: 0.4,
        presence: 0.3,
    });
}
