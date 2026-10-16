//! ANE generation through `generate_cached_ane_streaming`, on Qwen3-0.6B.
//!
//!     PMETAL_TEST_QWEN3_0_6B=<dir> \
//!     cargo test -p pmetal-models --features ane --release --test ane_generation -- --ignored --nocapture

#![cfg(all(target_os = "macos", feature = "ane"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use pmetal_models::{GenerationConfig, generate_cached_ane_streaming};

/// "The capital of France is"
const PROMPT: [u32; 5] = [785, 6722, 315, 9625, 374];

/// Greedy: `do_sample` off. (`temperature: 0.0` alone does not ask for it:
/// with `do_sample` on, a temperature at or below zero is taken as 1.)
fn generate(model: &Path) -> Vec<u32> {
    generate_with(model, GenerationConfig::greedy(8))
}

fn generate_with(model: &Path, config: GenerationConfig) -> Vec<u32> {
    generate_cached_ane_streaming(model, None, &PROMPT, &config, 512, |_| true)
        .expect("generate")
        .token_ids
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("PMETAL_TEST_QWEN3_0_6B").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
    }
    dir
}

/// A request is independent of the ones before it: the worker keeps one loaded
/// model, and its KV cache, positions and sampler history belong to the
/// request, not the model. Greedy generation is the case that shows it, since
/// the same prompt must give the same tokens however many requests came first.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn sequential_greedy_requests_are_identical() {
    let Some(model) = model_dir() else { return };
    let first = generate(&model);
    eprintln!("greedy: {first:?}");
    assert_eq!(first.len(), PROMPT.len() + 8);
    for i in 1..6 {
        assert_eq!(
            generate(&model),
            first,
            "request {i} differs from the first"
        );
    }
}

/// The ANE computes in fp16, so its greedy choice can differ from the GPU's
/// where two tokens are nearly tied. The rule: the ANE and the GPU agree on
/// every token up to the first difference, and at that step the GPU's own
/// logits put the ANE's token within [`NEAR_TIE`] of the token the GPU chose
/// (an fp16 rounding, not a different model). Past the difference the
/// continuations are different texts and are not compared.
const NEAR_TIE: f32 = 0.5;

#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn ane_greedy_agrees_with_the_gpu_up_to_a_near_tie() {
    use pmetal_models::DynamicModel;
    use pmetal_runtime::compat::{Array, Dtype};

    let Some(dir) = model_dir() else { return };
    let ane = generate(&dir);

    let mut gpu = DynamicModel::load(&dir).expect("load on the GPU");
    let logits_after = |gpu: &mut DynamicModel, ids: &[u32]| -> Vec<f32> {
        let ids: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let logits = gpu.forward(&input, None).expect("forward");
        let last = logits
            .index((.., ids.len() as i32 - 1, ..))
            .as_dtype(Dtype::Float32.as_i32());
        let n = last.size();
        let mut last = last;
        last.to_f32_vec(n).expect("logits")
    };

    for position in PROMPT.len()..ane.len() {
        let logits = logits_after(&mut gpu, &ane[..position]);
        let (gpu_token, gpu_best) =
            logits
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |best, (i, &v)| if v > best.1 { (i, v) } else { best },
                );
        let ane_token = ane[position] as usize;
        if gpu_token == ane_token {
            continue;
        }
        let gap = gpu_best - logits[ane_token];
        assert!(
            gap < NEAR_TIE,
            "at token {position} the ANE chose {ane_token} where the GPU's logits put {gpu_token} \
             {gap:.3} ahead: more than a near tie"
        );
        eprintln!("near tie at token {position}: ANE {ane_token}, GPU {gpu_token}, gap {gap:.3}");
        return;
    }
    pmetal_runtime::check_unobserved_error().unwrap();
}

/// A seeded sampled request reproduces, request after request.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn seeded_sampled_requests_reproduce() {
    let Some(model) = model_dir() else { return };
    let sampled = || GenerationConfig {
        max_new_tokens: 8,
        temperature: 0.8,
        seed: Some(1234),
        ..GenerationConfig::default()
    };
    let first = generate_with(&model, sampled());
    for i in 1..4 {
        assert_eq!(
            generate_with(&model, sampled()),
            first,
            "seeded request {i} differs from the first"
        );
    }
}

/// A server runs each request on whichever thread its pool hands out. The
/// model loads once, on the first request, and every later one, from any
/// thread and at the same time, reuses it.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn every_thread_shares_one_loaded_model() {
    let Some(model) = std::env::var_os("PMETAL_TEST_QWEN3_0_6B").map(PathBuf::from) else {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
        return;
    };
    let started = Instant::now();
    let first = generate(&model);
    let with_load = started.elapsed();

    let started = Instant::now();
    let others: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2).map(|_| s.spawn(|| generate(&model))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let reused = started.elapsed();
    eprintln!(
        "first request {with_load:.1?} (with the load); two more, from new threads, {reused:.1?}"
    );

    for other in &others {
        assert_eq!(other, &first);
    }
    assert!(
        reused * 4 < with_load,
        "two requests from other threads took {reused:.1?}, against {with_load:.1?} for the \
         first with its load: they loaded the model again"
    );
}
