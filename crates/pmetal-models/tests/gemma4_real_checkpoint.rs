//! A real Gemma 4 checkpoint against `transformers`, on both engines.
//!
//! The committed fixtures are tiny and synthetic, and blind to what only shows
//! at real weight magnitudes: GELU away from its linear region, the logit
//! softcap saturating, 48 layers of bf16 rounding. This runs a released
//! checkpoint's weights through the module-based port (what training and the
//! generic engines run) and through the native engine (what serves it), and
//! holds each to a reference dumped from `transformers` on the same token ids.
//!
//! Skipped unless `PMETAL_GEMMA4_REAL_DIR` (a checkpoint directory) and
//! `PMETAL_GEMMA4_REAL_REF` (a safetensors file holding `input_ids` `[1, T]`,
//! `logits` `[1, T, vocab]`, `final_hidden` `[1, T, hidden]` as f32, and
//! `argmax` `[T]`) are both set. The reference is a bf16 forward, so the
//! bounds are bf16 bounds, measured on the first run and stated below.

mod common;

use std::path::PathBuf;

use common::{load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::Array;
use pmetal_runtime::gemma4_native;

struct Real {
    dir: PathBuf,
    ids: Array,
    logits: Array,
    argmax: Vec<i32>,
}

fn real() -> Option<Real> {
    let dir = PathBuf::from(std::env::var_os("PMETAL_GEMMA4_REAL_DIR")?);
    let reference = PathBuf::from(std::env::var_os("PMETAL_GEMMA4_REAL_REF")?);
    let shard = load_shard(&reference);
    let argmax = pmetal_cache::test_utils::to_f32_vec_eval(ref_tensor(&shard, "argmax"))
        .into_iter()
        .map(|v| v as i32)
        .collect();
    Some(Real {
        dir,
        ids: ref_tensor(&shard, "input_ids").clone(),
        logits: ref_tensor(&shard, "logits").clone(),
        argmax,
    })
}

/// The largest logit difference allowed, relative to the largest logit.
/// Measured on the released checkpoints at full depth: 0.03 to 0.05 for both
/// engines (the bf16 rounding of 42 to 48 layers; logits reach the softcap of
/// 30). A truncated model is chaotic, since its routing and residual stream
/// are not trained to be stable past the layers it keeps, so
/// `PMETAL_GEMMA4_REAL_RTOL` widens the bound for it.
fn tolerance() -> Tolerance {
    let rtol = std::env::var("PMETAL_GEMMA4_REAL_RTOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1);
    Tolerance::new(0.0, rtol)
}

/// Agreement with the reference: every position's argmax, the cosine of every
/// position's logits, and the largest difference against the largest logit.
fn check(label: &str, got: &Array, real: &Real, tol: Tolerance, min_cosine: f32) {
    got.eval();
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    let report = ParityReport::compute_with_per_position(label, got, &real.logits, tol);
    print_report_table(std::slice::from_ref(&report));
    let worst_cosine = report
        .per_position_cosine
        .as_ref()
        .unwrap()
        .iter()
        .copied()
        .fold(1.0f32, f32::min);
    let argmax = argmax_last_axis(got);
    let flipped: Vec<usize> = argmax
        .iter()
        .zip(&real.argmax)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    println!(
        "{label}: worst per-position cosine {worst_cosine:.6}, argmax differs at {flipped:?} of {}",
        argmax.len()
    );
    assert!(report.passed(), "{label}: logits outside {tol:?}");
    assert!(worst_cosine >= min_cosine, "{label}: cosine {worst_cosine}");
    // bf16 can flip a near-tie, and only a near-tie: where the reference's two
    // best logits are within a bf16 step or two of each other. Measured on the
    // 12B checkpoint, the engines flip positions whose gap is 0.125 to 0.25,
    // against logits of magnitude 30 (a bf16 step there is 0.125 to 0.25).
    let vocab = real.logits.dim(2) as usize;
    let reference = pmetal_cache::test_utils::to_f32_vec_eval(&real.logits);
    for &position in &flipped {
        let mut row = reference[position * vocab..(position + 1) * vocab].to_vec();
        row.sort_by(|a, b| b.total_cmp(a));
        let gap = row[0] - row[1];
        assert!(
            gap < 0.5,
            "{label}: argmax flipped at position {position}, where the reference leads by {gap}"
        );
    }
}

#[test]
fn real_checkpoint_native_matches_transformers() {
    let Some(real) = real() else {
        eprintln!("PMETAL_GEMMA4_REAL_DIR / PMETAL_GEMMA4_REAL_REF unset; skipping");
        return;
    };
    let config = gemma4_native::load_config(&real.dir).expect("config");
    let weights = gemma4_native::load_model(&real.dir, &config).expect("native load");
    let mut cache = gemma4_native::build_cache(&weights, &config);
    let logits = gemma4_native::forward_step(&weights, &real.ids, &mut cache);
    check("native", &logits, &real, tolerance(), 0.99);
}

#[test]
fn real_checkpoint_module_port_matches_transformers() {
    let Some(real) = real() else {
        eprintln!("PMETAL_GEMMA4_REAL_DIR / PMETAL_GEMMA4_REAL_REF unset; skipping");
        return;
    };
    let mut model = DynamicModel::load(&real.dir).expect("module port loads it");
    let logits = model.forward(&real.ids, None).expect("forward");
    check("module port", &logits, &real, tolerance(), 0.99);
}
