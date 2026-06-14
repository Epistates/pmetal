//! Numerical-parity test for the *deterministic* DiffusionGemma sampler math.
//!
//! The discrete-diffusion generation loop is stochastic (a multinomial
//! proposal plus uniform renoising), but every transform the Rust port must
//! reproduce exactly is a pure function of the logits: the linear temperature
//! schedule, the Categorical token entropy, and the entropy-bound acceptance
//! mask. The dumper
//! (`.strategy/parity/dump_diffusion_gemma_reference.py --mode sampler`) runs
//! each on fixed seeded inputs from the transformers oracle; this test diffs
//! the Rust implementations against those references.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::Array;
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table, to_f32_vec_eval};

use pmetal_models::architectures::diffusion_gemma::{
    categorical_entropy, entropy_bound_accept, linear_temperature,
};

// Matches the dumper's `dump_sampler` constants.
const T_MIN: f32 = 0.4;
const T_MAX: f32 = 0.8;
const MAX_STEPS: i32 = 48;
const CUR_STEP: i32 = 24;
const ENTROPY_BOUND: f32 = 0.1;

fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    let pairs = pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"));
    pairs.into_iter().collect()
}

fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn i32_vec(a: &Array) -> Vec<i32> {
    to_f32_vec_eval(a).into_iter().map(|v| v as i32).collect()
}

#[test]
fn diffusion_gemma_sampler_parity() {
    let s = load_shard(&fixture_path(
        "diffusion_gemma_sampler_reference.safetensors",
    ));

    let raw_logits = ref_tensor(&s, "raw_logits");
    let processed_ref = ref_tensor(&s, "processed_logits");
    let mut reports = Vec::new();

    // 1. Linear temperature schedule (raw -> processed at cur_step).
    let processed_rust = linear_temperature(raw_logits, CUR_STEP, T_MIN, T_MAX, MAX_STEPS);
    reports.push(ParityReport::compute(
        "processed_logits",
        &processed_rust,
        processed_ref,
        Tolerance::new(1e-5, 1e-5),
    ));

    // 2. Categorical token entropy (random + peaked).
    let entropy_rust = categorical_entropy(processed_ref);
    reports.push(ParityReport::compute(
        "token_entropy",
        &entropy_rust,
        ref_tensor(&s, "token_entropy"),
        Tolerance::new(1e-4, 1e-4),
    ));
    let peaked_entropy_rust = categorical_entropy(ref_tensor(&s, "peaked_logits"));
    reports.push(ParityReport::compute(
        "peaked_entropy",
        &peaked_entropy_rust,
        ref_tensor(&s, "peaked_entropy"),
        Tolerance::new(1e-5, 1e-5),
    ));

    println!("\n== DiffusionGemma sampler parity report ==");
    print_report_table(&reports);

    // 3. Entropy-bound acceptance: the mask + accepted canvas are discrete, so
    // compare them exactly rather than via tolerance.
    let (accepted_rust, mask_rust) = entropy_bound_accept(
        processed_ref,
        ref_tensor(&s, "current_canvas"),
        ref_tensor(&s, "denoiser_canvas"),
        ENTROPY_BOUND,
    );
    let mask_rust_i: Vec<i32> = i32_vec(&mask_rust);
    let mask_ref_i = i32_vec(ref_tensor(&s, "accepted_token_mask"));
    let accepted_rust_i = i32_vec(&accepted_rust);
    let accepted_ref_i = i32_vec(ref_tensor(&s, "accepted_canvas"));
    println!("accepted_token_mask rust={mask_rust_i:?} ref={mask_ref_i:?}");
    println!("accepted_canvas     rust={accepted_rust_i:?} ref={accepted_ref_i:?}");

    assert_eq!(
        mask_rust_i, mask_ref_i,
        "entropy-bound acceptance mask mismatch"
    );
    assert_eq!(accepted_rust_i, accepted_ref_i, "accepted canvas mismatch");

    let failures: Vec<String> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "DiffusionGemma sampler parity failed at: {failures:?}"
    );
}
