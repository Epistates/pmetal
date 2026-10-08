//! Numerical parity for the Qwen3.5 family (`model_type = "qwen3_5"`: Qwen3.5,
//! 3.6 and 3.8 dense) against the authoritative Hugging Face `transformers`
//! `Qwen3_5ForConditionalGeneration` oracle, on both of pmetal's engines: the
//! `DynamicModel` path (`architectures/qwen3_next.rs`) and the native bridge
//! (`pmetal_bridge::qwen3_native`), which is what `pmetal infer` runs.
//!
//! Each fixture is a whole checkpoint directory, dumped by
//! `.strategy/parity/dump_qwen3_5_reference.py`: the nested `config.json` Qwen
//! ships, an fp32 `model.safetensors` under the released key names (vision
//! tower and bundled `mtp.*` predictor included), and the reference
//! activations. Both engines load it through their production loaders.
//!
//! Three profiles cover the config surface Qwen3.8 exercises:
//!
//! * `swish`: Qwen3.8-27B's text config shrunk. `output_gate_type: "swish"`,
//!   `layer_types` omitted so the layout comes from `full_attention_interval`,
//!   an untied head, partial rotary 0.25 under an interleaved mRoPE section.
//! * `sigmoid`: `output_gate_type: "sigmoid"`, an explicit `layer_types` that no
//!   interval reproduces, a head tied by the outer `tie_word_embeddings` while
//!   the text config says untied (transformers goes by the outer one; the
//!   checkpoint has no `lm_head.weight`), and a legacy top-level `rope_theta`
//!   that `rope_parameters` overrides.
//! * `yarn`: `swish` with the static YaRN block the Qwen3.8 card gives for
//!   long contexts (`factor` 4), shrunk to `original_max_position_embeddings`
//!   16 so the prompt runs past it.
//!
//! `vocab_size` differs from `hidden_size`, so a head applied untransposed is
//! a bridge error here, which every step drains, instead of a silently wrong
//! matrix.
//!
//! Every profile is checked on an uncached prefill and on a cached decode that
//! prefills part of the prompt and then feeds the rest one token at a time,
//! each step compared to the reference's row for that position. The sequence
//! is 70 tokens, past the reference's 64-token chunk boundary.
//!
//! The CPU hybrid engine (`pmetal_metal::ane::inference_hybrid`, which
//! `infer --ane` runs on a flat dense text config) is checked the same way:
//! it has only a single-token step, so its whole sequence is a cached decode.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::{Array, ops::slice_axis};
use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::qwen3_next_mtp::load_qwen3_next_mtp_from_dir;
use serial_test::serial;

/// Tokens fed as one prefill before the cached decode takes over.
const PREFILL: i32 = 40;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits
/// is 0.8-1.4e-5 (recorded in each fixture's meta), and every engine lands at
/// or under 1.2e-5 on every tap, step and profile. The gate is ~4x that: the
/// query/key epsilon this suite was written to catch moved the logits by
/// 3e-3, a wrong gate activation or conv state by O(1).
const PREFILL_TOL: Tolerance = Tolerance::new(5e-5, 0.0);
const DECODE_TOL: Tolerance = Tolerance::new(5e-5, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn path(&self) -> &Path {
        self._dir.path()
    }

    fn input_ids(&self) -> Array {
        ref_tensor(&self.reference, "input_ids").clone()
    }

    fn seq_len(&self) -> i32 {
        self.input_ids().dim(1)
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }
}

/// Lay the profile out as a checkpoint directory: `config.json` +
/// `model.safetensors`, nothing else.
fn checkpoint(profile: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("qwen3_5_{profile}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(&format!("qwen3_5_{profile}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let reference = load_shard(&fixture_path(&format!(
        "qwen3_5_{profile}_reference.safetensors"
    )));
    Fixture {
        _dir: dir,
        reference,
    }
}

/// Positions `[start, end)` of a `[1, T, D]` tensor.
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

fn assert_same_argmax(title: &str, got: &Array, want: &Array) {
    assert_eq!(
        argmax_last_axis(got),
        argmax_last_axis(want),
        "{title}: greedy tokens differ from the reference"
    );
}

// ---------------------------------------------------------------------------
// DynamicModel path
// ---------------------------------------------------------------------------

fn dynamic_prefill(profile: &str) {
    let fx = checkpoint(profile);
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let qwen = model
        .as_qwen3_next_mut()
        .expect("qwen3_5 routes to Qwen3Next");

    let taps = vec![0, 1, 2, 3];
    let mut capture = SpecCapture::with_layers_and_embedding(taps.clone(), false);
    let (hidden, logits) = qwen
        .forward_hidden_with_capture(&fx.input_ids(), None, None, None, &mut capture)
        .expect("forward runs");
    drain_bridge("dynamic prefill");

    let mut reports = Vec::new();
    for idx in taps {
        let got = capture.hidden_states.get(&idx).expect("tap captured");
        let name = format!("layer_{idx}_hidden");
        reports.push(ParityReport::compute(
            &name,
            got,
            &fx.tensor(&name),
            PREFILL_TOL,
        ));
    }
    reports.push(ParityReport::compute(
        "final_hidden",
        &hidden,
        &fx.tensor("final_hidden"),
        PREFILL_TOL,
    ));
    reports.push(ParityReport::compute_with_per_position(
        "logits",
        &logits,
        &fx.tensor("logits"),
        PREFILL_TOL,
    ));
    assert_all_pass(&format!("{profile}: dynamic prefill"), &reports);
    assert_same_argmax(profile, &logits, &fx.tensor("logits"));
}

fn dynamic_cached_decode(profile: &str) {
    let fx = checkpoint(profile);
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let t = fx.seq_len();
    let ids = fx.input_ids();
    let want = fx.tensor("logits");

    let mut kv = model.create_cache(t as usize + 1);
    let mut mamba = model
        .create_mamba_cache()
        .expect("hybrid model has a GDN cache");

    let prefill = model
        .forward_with_hybrid_cache(
            &rows(&ids, 0, PREFILL),
            None,
            Some(&mut kv),
            Some(&mut mamba),
        )
        .expect("cached prefill");
    drain_bridge("dynamic cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &prefill,
        &rows(&want, 0, PREFILL),
        DECODE_TOL,
    )];

    for pos in PREFILL..t {
        let step = model
            .forward_with_hybrid_cache(
                &rows(&ids, pos, pos + 1),
                None,
                Some(&mut kv),
                Some(&mut mamba),
            )
            .expect("decode step");
        drain_bridge("dynamic decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            DECODE_TOL,
        ));
    }
    assert_all_pass(&format!("{profile}: dynamic cached decode"), &reports);

    // The InlineArray decode path is tried on the first step only: the swish
    // checkpoint takes it, and the sigmoid one keeps the reason it can't
    // rather than trying, and logging, again every step.
    let DynamicModel::Qwen3Next(qwen) = &model else {
        panic!("{profile}: a Qwen3.5 checkpoint loads as Qwen3Next");
    };
    let inline = qwen.inline_weights.as_ref().map(Result::is_ok);
    let want_inline = Some(profile != "sigmoid");
    assert_eq!(inline, want_inline, "{profile}: InlineArray decode weights");
}

fn dynamic_mtp(profile: &str) {
    let fx = checkpoint(profile);
    let config_text = std::fs::read_to_string(fx.path().join("config.json")).unwrap();
    let model = DynamicModel::from_config(&config_text).expect("config builds");
    let DynamicModel::Qwen3Next(target) = model else {
        panic!("qwen3_5 routes to Qwen3Next");
    };
    let mut mtp = load_qwen3_next_mtp_from_dir(fx.path(), target.config()).expect("MTP head loads");

    let t = fx.seq_len();
    let next_ids = rows(&fx.input_ids(), 1, t);
    let hidden = rows(&fx.tensor("final_hidden"), 0, t - 1);
    let (_, logits) = mtp
        .forward_logits(&next_ids, &hidden, None, None, 0)
        .expect("MTP forward");
    drain_bridge("mtp forward");

    let reports = vec![ParityReport::compute_with_per_position(
        "mtp_logits",
        &logits,
        &fx.tensor("mtp_logits"),
        PREFILL_TOL,
    )];
    assert_all_pass(&format!("{profile}: MTP head"), &reports);
}

// ---------------------------------------------------------------------------
// Native bridge path (`pmetal infer`)
// ---------------------------------------------------------------------------

fn native_prefill_and_decode(profile: &str) {
    use pmetal_bridge::qwen3_native::{NativeCache, forward_step_hidden, load_config, load_model};

    let fx = checkpoint(profile);
    let config = load_config(fx.path()).expect("native config parses");
    let weights = load_model(fx.path(), &config).expect("native weights load");
    drain_bridge("native load");
    let t = fx.seq_len();
    let ids = fx.input_ids();
    let want = fx.tensor("logits");

    // Uncached prefill over the whole sequence.
    let mut cache = NativeCache::new_empty(&weights);
    let (hidden, logits) = forward_step_hidden(&weights, &ids, &mut cache);
    drain_bridge("native prefill");
    let reports = vec![
        ParityReport::compute(
            "final_hidden",
            &hidden,
            &fx.tensor("final_hidden"),
            PREFILL_TOL,
        ),
        ParityReport::compute_with_per_position("logits", &logits, &want, PREFILL_TOL),
    ];
    assert_all_pass(&format!("{profile}: native prefill"), &reports);
    assert_same_argmax(profile, &logits, &want);

    // Partial prefill, then one token at a time.
    let mut cache = NativeCache::new_empty(&weights);
    let (_, prefill) = forward_step_hidden(&weights, &rows(&ids, 0, PREFILL), &mut cache);
    drain_bridge("native cached prefill");
    let mut reports = vec![ParityReport::compute(
        "prefill_logits",
        &prefill,
        &rows(&want, 0, PREFILL),
        DECODE_TOL,
    )];
    for pos in PREFILL..t {
        let (_, step) = forward_step_hidden(&weights, &rows(&ids, pos, pos + 1), &mut cache);
        drain_bridge("native decode step");
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &step,
            &rows(&want, pos, pos + 1),
            DECODE_TOL,
        ));
    }
    assert_all_pass(&format!("{profile}: native cached decode"), &reports);
}

// ---------------------------------------------------------------------------
// CPU hybrid engine (`infer --ane` on a flat Qwen3.5 text config)
// ---------------------------------------------------------------------------

/// The CPU engine feeds every token through its single-token step, so its
/// logits for the whole sequence are a cached decode from the first token.
#[cfg(feature = "ane")]
fn hybrid_cpu_decode(profile: &str) {
    use pmetal_metal::ane::inference_hybrid::Qwen3NextInferenceEngine;

    let fx = checkpoint(profile);
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.path().join("config.json")).unwrap())
            .unwrap();
    let t = fx.seq_len();
    let config = pmetal_models::hybrid_cpu_inference_config(&config_json, t as usize)
        .expect("CPU hybrid config parses");
    let mut engine = Qwen3NextInferenceEngine::new(config).expect("engine builds");
    engine
        .load_weights_safetensors(fx.path())
        .expect("CPU hybrid weights load");

    let ids = fx.input_ids().as_type::<i32>();
    ids.eval();
    let ids: Vec<u32> = ids.as_slice::<i32>().iter().map(|&id| id as u32).collect();
    let rows = engine.prompt_logits(&ids).expect("CPU hybrid decodes");
    let vocab = rows[0].len() as i32;
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let logits = Array::from_slice(&flat, &[1, t, vocab]);

    let want = fx.tensor("logits");
    let reports = vec![ParityReport::compute_with_per_position(
        "logits", &logits, &want, DECODE_TOL,
    )];
    assert_all_pass(&format!("{profile}: CPU hybrid decode"), &reports);
    assert_same_argmax(profile, &logits, &want);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn swish_dynamic_prefill_matches_transformers() {
    dynamic_prefill("swish");
}

#[test]
#[serial]
fn swish_dynamic_cached_decode_matches_transformers() {
    dynamic_cached_decode("swish");
}

#[test]
#[serial]
fn swish_mtp_head_matches_reference() {
    dynamic_mtp("swish");
}

#[test]
#[serial]
fn swish_native_matches_transformers() {
    native_prefill_and_decode("swish");
}

#[test]
#[serial]
fn sigmoid_dynamic_prefill_matches_transformers() {
    dynamic_prefill("sigmoid");
}

#[test]
#[serial]
fn sigmoid_dynamic_cached_decode_matches_transformers() {
    dynamic_cached_decode("sigmoid");
}

#[test]
#[serial]
fn sigmoid_mtp_head_matches_reference() {
    dynamic_mtp("sigmoid");
}

#[test]
#[serial]
fn sigmoid_native_matches_transformers() {
    native_prefill_and_decode("sigmoid");
}

#[cfg(feature = "ane")]
#[test]
#[serial]
fn swish_cpu_hybrid_matches_transformers() {
    hybrid_cpu_decode("swish");
}

#[cfg(feature = "ane")]
#[test]
#[serial]
fn sigmoid_cpu_hybrid_matches_transformers() {
    hybrid_cpu_decode("sigmoid");
}

// Static YaRN (`rope_parameters.rope_type: "yarn"`), the long-context setting
// the Qwen3.8 card documents: blended frequencies, and the attention factor on
// the rotated quarter of each head only. The 70-token prompt runs past
// `original_max_position_embeddings` (16), and the cached decode continues
// there, through the fused kernel's explicit periods.

#[test]
#[serial]
fn yarn_dynamic_prefill_matches_transformers() {
    dynamic_prefill("yarn");
}

#[test]
#[serial]
fn yarn_dynamic_cached_decode_matches_transformers() {
    dynamic_cached_decode("yarn");
}

#[test]
#[serial]
fn yarn_mtp_head_matches_reference() {
    dynamic_mtp("yarn");
}

#[test]
#[serial]
fn yarn_native_matches_transformers() {
    native_prefill_and_decode("yarn");
}
