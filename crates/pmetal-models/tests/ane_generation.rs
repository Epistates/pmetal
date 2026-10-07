//! ANE generation through `generate_cached_ane_streaming`, on Qwen3-0.6B.
//!
//!     PMETAL_TEST_QWEN3_0_6B=<dir> \
//!     cargo test -p pmetal-models --features ane --release --test ane_generation -- --ignored --nocapture

#![cfg(all(target_os = "macos", feature = "ane"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use pmetal_models::{GenerationConfig, generate_cached_ane_streaming};

fn generate(model: &Path) -> Vec<u32> {
    let config = GenerationConfig {
        max_new_tokens: 8,
        temperature: 0.0,
        ..GenerationConfig::default()
    };
    // "The capital of France is"
    generate_cached_ane_streaming(
        model,
        None,
        &[785, 6722, 315, 9625, 374],
        &config,
        512,
        |_| true,
    )
    .expect("generate")
    .token_ids
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
