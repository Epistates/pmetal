//! Llama 4 text against the authoritative Hugging Face `transformers`
//! `Llama4ForCausalLM`, on both of pmetal's engines, over a whole prompt and
//! through the caches one token at a time.
//!
//! The fixture (`.strategy/parity/dump_llama4_text_reference.py`) is a whole
//! four-layer checkpoint: Llama 3 frequency bands shrunk so an 8-wide head's
//! four frequencies land in all three bands, interleaved RoPE and the
//! weightless QK-norm on layers 0-2, NoPE on layer 3 with attention
//! temperature tuning at `floor_scale` 4, so its query scale steps every four
//! positions. Every layer is dense, which isolates attention.
//!
//! A cached decode feeds token `p` alone and must reproduce row `p` of the
//! uncached forward: that holds only if new keys rotate at their absolute
//! position and the NoPE layer scales queries by it, not by their index in
//! the call.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, Dtype, ops::slice_axis};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over.
const PREFILL: i32 = 7;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits
/// is 2.7e-7 (recorded in the fixture's meta).
const TOL: Tolerance = Tolerance::new(2e-5, 0.0);

struct Fixture {
    dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }
}

fn checkpoint() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("llama4_text_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path("llama4_text_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    Fixture {
        dir,
        reference: load_shard(&fixture_path("llama4_text_reference.safetensors")),
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = a.subtract(b).abs();
    let d = (0..d.ndim()).fold(d, |d, _| d.max_axis(-1, false));
    d.item_f32()
}

fn assert_all_pass(title: &str, reports: &[ParityReport]) {
    println!("\n== {title} ==");
    print_report_table(reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{title}: {failed:?} out of tolerance");
}

/// One report per cached step: the prefill's rows, then each token's row.
fn step_reports(steps: &[Array], want: &Array) -> Vec<ParityReport> {
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &steps[0],
        &rows(want, 0, PREFILL),
        TOL,
    )];
    for (i, step) in steps[1..].iter().enumerate() {
        let pos = PREFILL + i as i32;
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            step,
            &rows(want, pos, pos + 1),
            TOL,
        ));
    }
    reports
}

/// The `DynamicModel` path (`architectures/llama4.rs`), which training,
/// LoRA and `serve` run. It rotated every forward from position 0 and kept
/// no cache, so a cached decode attended to nothing but the new token.
#[test]
#[serial]
fn dynamic_llama4_matches_transformers() {
    use pmetal_models::DynamicModel;

    let fx = checkpoint();
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let logits = model.forward(&ids, None).expect("forward");
    drain_bridge("dynamic forward");
    assert_all_pass(
        "dynamic forward",
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    // Explicit positions 0..t are the contiguous run.
    let positions = Array::from_iter(0..t, &[t]).as_dtype(Dtype::Int32.as_i32());
    let explicit = model
        .forward_with_positions(&ids, None, Some(&positions))
        .expect("forward with positions");
    drain_bridge("dynamic forward with positions");
    assert!(max_abs_diff(&explicit, &logits) <= 1e-5);

    let mut cache = model.create_cache(t as usize + 1);
    let mut steps = vec![
        model
            .forward_with_cache(&rows(&ids, 0, PREFILL), None, Some(&mut cache))
            .expect("cached prefill"),
    ];
    for pos in PREFILL..t {
        steps.push(
            model
                .forward_with_cache(&rows(&ids, pos, pos + 1), None, Some(&mut cache))
                .expect("decode step"),
        );
        drain_bridge("dynamic decode step");
    }
    assert_all_pass("dynamic cached decode", &step_reports(&steps, &want));
}

/// The native engine (`pmetal_bridge::llama4_native`): its compiled
/// single-token graph rotates by the bands' period table, and must give what
/// the per-op path gives and what transformers gives.
#[test]
#[serial]
fn native_llama4_matches_transformers() {
    use pmetal_bridge::llama4_native::{NativeCache, forward_step, load_config, load_model};

    let fx = checkpoint();
    let config = load_config(fx.path()).expect("native config parses");
    let weights = load_model(fx.path(), &config).expect("native weights load");
    drain_bridge("native load");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let mut cache = NativeCache::new_empty(&weights);
    let logits = forward_step(&weights, &ids, &mut cache);
    drain_bridge("native prefill");
    assert_all_pass(
        "native prefill",
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );

    let decode = |compiled: bool| {
        pmetal_bridge::decode::set_compiled_decode(compiled);
        let mut cache = NativeCache::new_empty(&weights);
        let mut steps = vec![forward_step(&weights, &rows(&ids, 0, PREFILL), &mut cache)];
        for pos in PREFILL..t {
            steps.push(forward_step(
                &weights,
                &rows(&ids, pos, pos + 1),
                &mut cache,
            ));
            drain_bridge("native decode step");
        }
        pmetal_bridge::decode::set_compiled_decode(true);
        steps
    };
    let compiled = decode(true);
    let per_op = decode(false);
    assert_all_pass("native cached decode", &step_reports(&compiled, &want));
    let graph_vs_ops = compiled
        .iter()
        .zip(&per_op)
        .map(|(a, b)| max_abs_diff(a, b))
        .fold(0f32, f32::max);
    println!("compiled decode vs per-op decode: max |diff| {graph_vs_ops:e}");
    assert!(
        graph_vs_ops <= 1e-5,
        "the compiled decode graph drifted from the per-op path by {graph_vs_ops:e}"
    );
}
