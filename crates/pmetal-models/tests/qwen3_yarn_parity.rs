//! Dense Qwen3 with YaRN against the authoritative Hugging Face
//! `transformers` `Qwen3ForCausalLM`, on both of pmetal's engines.
//!
//! The fixture (`.strategy/parity/dump_qwen3_yarn_reference.py`) is a whole
//! checkpoint directory whose `config.json` carries YaRN the way the Qwen3 card
//! documents it, a legacy `rope_scaling` block with `factor` 4, shrunk from
//! `original_max_position_embeddings` 32768 to 64 so the 96-token prompt runs
//! well past the original window. It also carries the stray `attn_factor` key
//! DeepSeek-R1-0528-Qwen3-8B ships, which transformers ignores: the attention
//! factor is `0.1 ln(4) + 1` from `factor`.
//!
//! Each engine is checked on an uncached prefill of the whole prompt and on a
//! cached decode that prefills the first 40 tokens (inside the original
//! window) and feeds the rest one at a time, every step compared to the
//! reference's row for that position. YaRN changes the frequencies at every
//! position, not only past the window, so the prefill alone already fails
//! without it; the decode is what checks the cache rotates new keys at their
//! absolute positions with the same table.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, ops::slice_axis};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over: inside the
/// 64-token original window, so the decode crosses it.
const PREFILL: i32 = 40;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits
/// is 2.1e-6 (recorded in the fixture's meta). Dropping the attention factor
/// moves them by 1e-2, plain RoPE in place of YaRN by more.
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
        fixture_path("qwen3_yarn_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path("qwen3_yarn_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    Fixture {
        dir,
        reference: load_shard(&fixture_path("qwen3_yarn_reference.safetensors")),
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
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

/// The `DynamicModel` path (`architectures/qwen3.rs`), which training,
/// LoRA and `serve` run. Its "yarn" used to be an NTK-style base change plus
/// a 1/4 position scale, wrong from the first token.
#[test]
#[serial]
fn dynamic_qwen3_yarn_matches_transformers() {
    use pmetal_models::DynamicModel;

    let fx = checkpoint();
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let logits = model.forward(&ids, None).expect("prefill");
    drain_bridge("dynamic prefill");
    assert_all_pass(
        "dynamic prefill",
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    let mut cache = model.create_cache(t as usize + 1);
    let prefill = model
        .forward_with_cache(&rows(&ids, 0, PREFILL), None, Some(&mut cache))
        .expect("cached prefill");
    drain_bridge("dynamic cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &prefill,
        &rows(&want, 0, PREFILL),
        TOL,
    )];
    for pos in PREFILL..t {
        let step = model
            .forward_with_cache(&rows(&ids, pos, pos + 1), None, Some(&mut cache))
            .expect("decode step");
        drain_bridge("dynamic decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass("dynamic cached decode", &reports);
    // Continuous batching keeps a scaled RoPE on the serial path.
    assert!(!model.supports_fused_batched());
}

#[test]
#[serial]
fn native_qwen3_yarn_matches_transformers() {
    use pmetal_bridge::qwen3_native::{NativeCache, forward_step_hidden, load_config, load_model};

    let fx = checkpoint();
    let config = load_config(fx.path()).expect("native config parses");
    assert!(config.is_qwen3_dense());
    let rope = config.scaled_rope().expect("the config's YaRN is read");
    assert_eq!(rope.rotary().scaling.rope_type(), "yarn");
    assert!((rope.attention_factor() as f64 - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-6);
    let weights = load_model(fx.path(), &config).expect("native weights load");
    drain_bridge("native load");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let mut cache = NativeCache::new_empty(&weights);
    let (hidden, logits) = forward_step_hidden(&weights, &ids, &mut cache);
    drain_bridge("native prefill");
    assert_all_pass(
        "native prefill",
        &[
            ParityReport::compute("final_hidden", &hidden, &fx.tensor("final_hidden"), TOL),
            ParityReport::compute_with_per_position("logits", &logits, &want, TOL),
        ],
    );
    assert_eq!(argmax_last_axis(&logits), argmax_last_axis(&want));

    let mut cache = NativeCache::new_empty(&weights);
    let (_, prefill) = forward_step_hidden(&weights, &rows(&ids, 0, PREFILL), &mut cache);
    drain_bridge("native cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &prefill,
        &rows(&want, 0, PREFILL),
        TOL,
    )];
    for pos in PREFILL..t {
        let (_, step) = forward_step_hidden(&weights, &rows(&ids, pos, pos + 1), &mut cache);
        drain_bridge("native decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass("native cached decode", &reports);
}
