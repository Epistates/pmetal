//! The native gpt-oss engine (`pmetal_bridge::gpt_oss_native`) against the
//! authoritative Hugging Face `transformers` `GptOssForCausalLM`.
//!
//! The fixture (`.strategy/parity/dump_gpt_oss_native_reference.py`) is a
//! whole two-layer checkpoint with what the release carries: per-head
//! attention sinks seeded non-zero, YaRN with `truncate: false`, a sliding
//! window of 4 on layer 0 beside full attention on layer 1, the biased top-k
//! router with its softmax over the chosen logits, and the clamped GLU on
//! interleaved gate/up columns. It ships the same weights in transformers'
//! layout and in the stacked layout MLX conversions use; both must load to
//! the same model.
//!
//! Each layout is checked on an uncached prefill of the 12-token prompt, on a
//! cached decode that prefills 3 tokens and feeds the other 9 one at a time
//! (crossing the window, through the compiled single-token graph and through
//! the per-op path), and on a decode whose second prefill chunk starts past
//! the window, so the ring has wrapped and the band mask is not plain causal.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::{Array, ops::slice_axis};
use pmetal_bridge::gpt_oss_native::{
    NativeCache, NativeWeights, forward_step, load_config, load_model,
};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use serial_test::serial;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits
/// is 2.0e-7 (recorded in the fixture's meta). Dropping the sinks, the
/// window or the router's softmax moves them by 1e-3 or more.
const TOL: Tolerance = Tolerance::new(2e-5, 0.0);

struct Fixture {
    _dir: tempfile::TempDir,
    weights: NativeWeights,
    reference: HashMap<String, Array>,
}

impl Fixture {
    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }
}

/// The `<model>_config.json` / `<model>_reference.safetensors` fixture with
/// `weights_file` as its checkpoint.
fn checkpoint(model: &str, weights_file: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("{model}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(weights_file),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let weights = load(dir.path());
    Fixture {
        _dir: dir,
        weights,
        reference: load_shard(&fixture_path(&format!("{model}_reference.safetensors"))),
    }
}

fn load(path: &Path) -> NativeWeights {
    let config = load_config(path).expect("config parses");
    let weights = load_model(path, &config).expect("weights load");
    drain_bridge("load");
    weights
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

/// Feed the prompt in `chunks` (lengths summing to the prompt), one cache
/// throughout, and return each chunk's logits.
fn run_chunks(fx: &Fixture, chunks: &[i32], compiled: bool) -> Vec<(i32, InlineArray)> {
    pmetal_bridge::decode::set_compiled_decode(compiled);
    let ids = fx.tensor("input_ids");
    let mut cache = NativeCache::new_empty(&fx.weights);
    let mut start = 0;
    let mut out = Vec::new();
    for &len in chunks {
        let logits = forward_step(&fx.weights, &rows(&ids, start, start + len), &mut cache);
        drain_bridge("forward step");
        out.push((start, logits));
        start += len;
        if start == chunks[0] {
            // As generation does between the prefill and the first decode.
            cache.eval_and_detach_states();
        }
    }
    pmetal_bridge::decode::set_compiled_decode(true);
    out
}

fn check_layout(model: &str, weights_file: &str) {
    let fx = checkpoint(model, weights_file);
    let want = fx.tensor("logits");
    let t = want.dim(1);

    // Uncached prefill of the whole prompt.
    let whole = run_chunks(&fx, &[t], true);
    let logits = &whole[0].1;
    assert_all_pass(
        &format!("{weights_file}: prefill"),
        &[ParityReport::compute_with_per_position(
            "logits", logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(logits), argmax_last_axis(&want));

    // Prefill 3, then one token at a time across the window of 4, through
    // the compiled graph and through the per-op path.
    let mut chunks = vec![3];
    chunks.extend(std::iter::repeat_n(1, (t - 3) as usize));
    let compiled = run_chunks(&fx, &chunks, true);
    let per_op = run_chunks(&fx, &chunks, false);
    let mut reports = Vec::new();
    let mut graph_vs_ops = 0f32;
    for ((start, step), (_, op_step)) in compiled.iter().zip(&per_op) {
        let len = step.dim(1);
        graph_vs_ops = graph_vs_ops.max(max_abs_diff(step, op_step));
        reports.push(ParityReport::compute(
            &format!("pos_{start}"),
            step,
            &rows(&want, *start, start + len),
            TOL,
        ));
    }
    assert_all_pass(&format!("{weights_file}: cached decode"), &reports);
    println!("compiled decode vs per-op decode: max |diff| {graph_vs_ops:e}");
    assert!(
        graph_vs_ops <= 1e-5,
        "the compiled decode graph drifted from the per-op path by {graph_vs_ops:e}"
    );

    // A second prefill chunk that starts past the window: the ring has
    // wrapped, and the chunk's queries need the band mask.
    let reports: Vec<_> = run_chunks(&fx, &[6, 4, 1, 1], true)
        .iter()
        .map(|(start, logits)| {
            ParityReport::compute(
                &format!("chunk_at_{start}"),
                logits,
                &rows(&want, *start, start + logits.dim(1)),
                TOL,
            )
        })
        .collect();
    assert_all_pass(&format!("{weights_file}: chunked prefill"), &reports);
}

#[test]
#[serial]
fn native_gpt_oss_matches_transformers_from_the_transformers_layout() {
    check_layout("gpt_oss_native", "gpt_oss_native_weights.safetensors");
}

#[test]
#[serial]
fn native_gpt_oss_matches_transformers_from_the_stacked_layout() {
    check_layout(
        "gpt_oss_native",
        "gpt_oss_native_stacked_weights.safetensors",
    );
}

/// The release's layout: experts as MXFP4 `*_blocks` / `*_scales` (random
/// here, hidden and intermediate 64), everything else dense. They stay
/// packed and run through `gather_qmm`; transformers dequantizes them
/// (`convert_moe_packed_tensors`) for its reference.
#[test]
#[serial]
fn native_gpt_oss_runs_the_release_mxfp4_experts_packed() {
    check_layout("gpt_oss_mxfp4", "gpt_oss_mxfp4_weights.safetensors");
    let fx = checkpoint("gpt_oss_mxfp4", "gpt_oss_mxfp4_weights.safetensors");
    assert!(
        fx.weights.experts_packed(),
        "the MXFP4 experts were unpacked at load"
    );
}

/// The `DynamicModel` path (training, LoRA, `serve`) on the same checkpoint
/// directory: its loader reads the fused experts the native engine reads,
/// packed when the checkpoint is, and the model must match transformers on
/// an uncached forward and a cached decode across the window.
fn check_dynamic(model: &str, weights_file: &str) {
    use pmetal_models::DynamicModel;

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("{model}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(weights_file),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let reference = load_shard(&fixture_path(&format!("{model}_reference.safetensors")));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let want = ref_tensor(&reference, "logits").clone();
    let t = ids.dim(1);

    let mut dynamic = DynamicModel::load(dir.path()).expect("checkpoint loads");
    drain_bridge("dynamic load");
    let logits = dynamic.forward(&ids, None).expect("forward");
    drain_bridge("dynamic forward");
    assert_all_pass(
        &format!("{weights_file}: dynamic forward"),
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );

    let mut cache = dynamic.create_cache(t as usize + 1);
    let mut reports = Vec::new();
    let mut start = 0;
    for len in std::iter::once(3).chain(std::iter::repeat_n(1, (t - 3) as usize)) {
        let step = dynamic
            .forward_with_cache(&rows(&ids, start, start + len), None, Some(&mut cache))
            .expect("cached step");
        drain_bridge("dynamic cached step");
        reports.push(ParityReport::compute(
            &format!("pos_{start}"),
            &step,
            &rows(&want, start, start + len),
            TOL,
        ));
        start += len;
    }
    assert_all_pass(&format!("{weights_file}: dynamic cached decode"), &reports);
}

#[test]
#[serial]
fn dynamic_gpt_oss_matches_transformers_from_the_transformers_layout() {
    check_dynamic("gpt_oss_native", "gpt_oss_native_weights.safetensors");
}

#[test]
#[serial]
fn dynamic_gpt_oss_runs_the_release_mxfp4_experts_packed() {
    check_dynamic("gpt_oss_mxfp4", "gpt_oss_mxfp4_weights.safetensors");
}

/// A loss differentiated through the packed experts (`gather_qmm`) has to
/// reach everything below them, as training adapters there needs: the input
/// gradient is finite, non-zero, and the one the same experts give unpacked.
#[test]
#[serial]
fn packed_experts_pass_the_gradient_to_their_input() {
    use pmetal_models::DynamicModel;

    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("gpt_oss_mxfp4_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path("gpt_oss_mxfp4_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let mut dynamic = DynamicModel::load(dir.path()).expect("checkpoint loads");
    let DynamicModel::GptOss(model) = &mut dynamic else {
        panic!("a gpt-oss checkpoint loads as GptOss");
    };
    let moe = &mut model.model.layers[0].mlp;
    let packed = moe.fused_experts().expect("fused experts").clone();
    assert!(packed.is_packed());
    let x = pmetal_bridge::compat::random::normal(&[5, 64], pmetal_bridge::compat::Dtype::Float32);

    let grad_of = |moe: &mut pmetal_models::architectures::gpt_oss::GptOssMoE| {
        pmetal_bridge::clear_last_error();
        let (loss, grads) = pmetal_bridge::compat::nn::value_and_grad_explicit(
            |arrays: &[Array]| moe.forward(&arrays[0]).unwrap().square().sum(None),
            std::slice::from_ref(&x),
            &[],
        )
        .unwrap();
        drain_bridge("value_and_grad");
        assert!(loss.item::<f32>().is_finite());
        grads[0].eval();
        drain_bridge("gradient");
        grads[0].clone()
    };
    let through_packed = grad_of(moe);
    moe.set_fused_experts(packed.unpacked(pmetal_bridge::compat::Dtype::Float32.as_i32()));
    let through_dense = grad_of(moe);

    let dx = through_packed.as_slice::<f32>();
    assert!(dx.iter().all(|g| g.is_finite()), "finite input gradient");
    assert!(dx.iter().any(|g| *g != 0.0), "non-zero input gradient");
    let diff = max_abs_diff(&through_packed, &through_dense);
    assert!(
        diff <= 1e-5,
        "packed and dense experts' gradients differ by {diff:e}"
    );
}

/// An MLX quantization packs every projection, which this engine doesn't
/// read: loading one is refused by name, not decoded as noise.
#[test]
#[serial]
fn native_gpt_oss_refuses_an_mlx_quantization_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("gpt_oss_native_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let scales = InlineArray::zeros(&[64, 1], pmetal_bridge::compat::Dtype::Float32.as_i32());
    InlineArray::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &[("model.layers.0.self_attn.q_proj.scales", &scales)],
    );
    let config = load_config(dir.path()).expect("config parses");
    let err = load_model(dir.path(), &config).expect_err("MLX-quantized weights are refused");
    assert!(err.contains("q_proj.scales"), "{err}");
    drain_bridge("refusal");
}
