//! A seeded generation picks the same tokens every time, on both engines.
//!
//! The seed used to reach the decode loop only as a reseed of the thread's
//! MLX key, and on the native engine (what `pmetal infer` runs for Qwen3,
//! Qwen3.5, Llama 4, DeepSeek, gpt-oss and Gemma 4) not at all. Each draw now
//! gets its own key from the seed and the token's index.
//!
//! The model is the tiny Qwen3.5 checkpoint the parity suite ships, sampled
//! at temperature 1 so every token is a real draw. Each case generates with
//! seed 5, then again on another thread after unseeded draws have moved that
//! thread's MLX key, and with seed 6.

use std::path::PathBuf;

use pmetal_bridge::compat::random;
use pmetal_bridge::decode::SamplingParams;
use pmetal_models::DynamicModel;
use pmetal_models::generation::{GenerationConfig, generate_cached_async_streaming};

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

/// Move the calling thread's MLX key the way unrelated work would.
fn draw_unseeded() {
    let _ = random::normal(&[64], pmetal_bridge::compat::Dtype::Float32).as_slice::<f32>()[0];
}

/// Each case undisturbed, then on a fresh thread with unseeded draws
/// between its tokens (`disturb`).
fn here_and_elsewhere(run: fn(Option<u64>, bool) -> Vec<u32>) {
    let first = run(Some(5), false);
    assert_eq!(first.len(), NEW_TOKENS);
    let again = std::thread::spawn(move || {
        random::seed(1234);
        run(Some(5), true)
    })
    .join()
    .unwrap();
    assert_eq!(first, again, "seed 5 drew different tokens on a second run");
    assert_ne!(
        first,
        run(Some(6), false),
        "seeds 5 and 6 drew the same tokens"
    );
    pmetal_bridge::check_unobserved_error().expect("no bridge op failed");
}

fn native(seed: Option<u64>, disturb: bool) -> Vec<u32> {
    use pmetal_bridge::qwen3_native::{NativeCache, generate, load_config, load_model};
    let dir = checkpoint();
    let config = load_config(dir.path()).expect("config");
    let weights = load_model(dir.path(), &config).expect("weights");
    let params = SamplingParams {
        seed,
        ..SamplingParams::new(1.0)
    };
    let mut cache = NativeCache::new_empty(&weights);
    let first =
        pmetal_bridge::qwen3_native::prefill_first_token(&weights, &mut cache, &PROMPT, &params);
    if disturb {
        draw_unseeded();
    }
    let (rest, _) = generate(&weights, &mut cache, first, NEW_TOKENS - 1, params, |_| {
        if disturb {
            draw_unseeded();
        }
        true
    });
    std::iter::once(first).chain(rest).collect()
}

fn dynamic(seed: Option<u64>, disturb: bool) -> Vec<u32> {
    let dir = checkpoint();
    let mut model = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let mut cache = model.create_cache(PROMPT.len() + NEW_TOKENS + 8);
    let mut mamba = model.create_mamba_cache();
    let mut config = GenerationConfig::sampling(NEW_TOKENS, 1.0);
    config.seed = seed;
    let out = generate_cached_async_streaming(
        |input, cache| model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut()),
        &PROMPT,
        config,
        &mut cache,
        |_| {
            if disturb {
                draw_unseeded();
            }
            true
        },
    )
    .expect("generate");
    out.token_ids[PROMPT.len()..].to_vec()
}

#[test]
fn the_native_engine_repeats_a_seeded_generation() {
    here_and_elsewhere(native);
}

#[test]
fn the_dynamic_model_path_repeats_a_seeded_generation() {
    here_and_elsewhere(dynamic);
}
