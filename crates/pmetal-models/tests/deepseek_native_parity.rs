//! The native DeepSeek engine (`pmetal_bridge::deepseek_native`) against the
//! authoritative Hugging Face `transformers` `DeepseekV3ForCausalLM` and
//! `DeepseekV2ForCausalLM`.
//!
//! Each fixture (transformers 5.19, torch 2.14.1) is a whole three-layer
//! checkpoint in the layout the releases ship: a dense first layer, then MoE
//! layers with every routed expert as its own `experts.E.*_proj.weight`, a
//! shared expert, one `kv_b_proj` per layer, YaRN with `mscale` 2 and
//! `mscale_all_dim` 1, and a trailing multi-token-prediction layer the engine
//! must ignore.
//!
//! * `v3`: sigmoid scoring, the bias-corrected group-limited top-k (16 experts
//!   in 4 groups, 2 groups kept, 4 experts chosen), normalised and scaled
//!   weights, low-rank Q.
//! * `v2_greedy` / `v2_group_limited_greedy`: softmax scoring, unnormalised
//!   weights, a direct Q projection; the plain top-k, and the group-max one.
//!
//! Each is checked on an uncached prefill of the 12-token prompt and on a
//! cached decode that prefills 3 tokens and feeds the rest one at a time, and
//! on prefill chunks that start past position 0 (a causal mask with an
//! offset).

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::InlineArray;
use pmetal_bridge::compat::{Array, ops::slice_axis};
use pmetal_bridge::deepseek_native::{
    NativeCache, NativeWeights, forward_step, load_config, load_model,
};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use serial_test::serial;

/// fp32 against fp32. The reference's own fp32-vs-fp64 noise on these logits
/// is under 1e-6 (recorded in the fixtures' meta); a wrong mask, a missed
/// group, a swapped score or a dropped mscale moves them by 1e-2.
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

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

fn checkpoint(variant: &str, weights_file: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("deepseek_native_{variant}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(weights_file),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let config = load_config(dir.path()).expect("config parses");
    let weights = load_model(dir.path(), &config).expect("weights load");
    drain_bridge("load");
    Fixture {
        _dir: dir,
        weights,
        reference: load_shard(&fixture_path(&format!(
            "deepseek_native_{variant}_reference.safetensors"
        ))),
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
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

/// Feed the prompt in `chunks` (lengths summing to the prompt) through one
/// cache and return each chunk's start and logits.
fn run_chunks(fx: &Fixture, chunks: &[i32]) -> Vec<(i32, InlineArray)> {
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
    out
}

fn check(variant: &str, weights_file: &str) {
    let fx = checkpoint(variant, weights_file);
    let want = fx.tensor("logits");
    let t = want.dim(1);

    let whole = run_chunks(&fx, &[t]);
    let logits = &whole[0].1;
    assert_all_pass(
        &format!("{variant}: prefill"),
        &[ParityReport::compute_with_per_position(
            "logits", logits, &want, TOL,
        )],
    );
    assert_eq!(argmax_last_axis(logits), argmax_last_axis(&want));

    let mut one_by_one = vec![3];
    one_by_one.extend(std::iter::repeat_n(1, (t - 3) as usize));
    for (title, chunks) in [
        ("cached decode", one_by_one),
        ("chunked prefill", vec![5, 4, 3]),
    ] {
        let reports: Vec<_> = run_chunks(&fx, &chunks)
            .iter()
            .map(|(start, logits)| {
                ParityReport::compute(
                    &format!("pos_{start}"),
                    logits,
                    &rows(&want, *start, start + logits.dim(1)),
                    TOL,
                )
            })
            .collect();
        assert_all_pass(&format!("{variant}: {title}"), &reports);
    }
}

/// An MLX-style 4-bit quantization of the V3 fixture (group 32, the router
/// gates at 8 bits, the experts stacked as `switch_mlp.*`), and the same
/// weights dequantized to dense in the same layout. The packed checkpoint has
/// to give the dense one's logits: embeddings, `lm_head`, every projection
/// that divides into groups, `kv_b_proj` (dequantized at load and split), the
/// gates and the stacked experts all read packed.
///
/// Weights whose input width does not divide into groups (the 48-wide
/// `q_b_proj`, the 16-wide expert `down_proj`) stay dense in both, as a
/// mixed-precision checkpoint leaves them.
#[test]
#[serial]
fn native_deepseek_runs_an_mlx_quantized_checkpoint() {
    const GROUP: i32 = 32;
    let source = load_shard(&fixture_path("deepseek_native_v3_weights.safetensors"));
    let mut packed: Vec<(String, Array)> = Vec::new();
    let mut dense: Vec<(String, Array)> = Vec::new();
    let mut overrides = serde_json::Map::new();

    // Quantize `w` (`[.., out, in]`) as module `module`, filing the packed
    // tensors and their dequantization.
    let mut quantize = |module: &str, w: &Array, bits: i32| {
        let (q, scales, biases) = w.quantize_weights(GROUP, bits);
        let deq = q.dequantize(&scales, &biases, GROUP, bits);
        packed.push((format!("{module}.weight"), q));
        packed.push((format!("{module}.scales"), scales));
        packed.push((format!("{module}.biases"), biases));
        dense.push((format!("{module}.weight"), deq));
        if bits != 4 {
            overrides.insert(
                module.to_string(),
                serde_json::json!({"group_size": GROUP, "bits": bits}),
            );
        }
    };

    let mut experts: std::collections::BTreeMap<(String, String), Vec<(usize, Array)>> =
        Default::default();
    let mut keys: Vec<&String> = source.keys().collect();
    keys.sort();
    let mut both = Vec::new();
    for key in keys {
        let w = &source[key];
        let module = key.strip_suffix(".weight");
        let divides = w.ndim() == 2 && w.dim(1) % GROUP == 0;
        if let Some((layer, rest)) = key.split_once(".mlp.experts.") {
            // `model.layers.L` / `E.proj.weight`
            let (e, proj) = rest.split_once('.').unwrap();
            let proj = proj.strip_suffix(".weight").unwrap();
            experts
                .entry((layer.to_string(), proj.to_string()))
                .or_default()
                .push((e.parse().unwrap(), w.clone()));
        } else if let (Some(module), true) = (module, divides) {
            let bits = if module.ends_with(".mlp.gate") { 8 } else { 4 };
            quantize(module, w, bits);
        } else {
            both.push((key.clone(), w.clone()));
        }
    }
    for ((layer, proj), mut parts) in experts {
        parts.sort_by_key(|(e, _)| *e);
        let stack = pmetal_bridge::compat::ops::stack_axis(
            &parts.iter().map(|(_, w)| w.clone()).collect::<Vec<_>>(),
            0,
        );
        let module = format!("{layer}.mlp.switch_mlp.{proj}");
        if stack.dim(2) % GROUP == 0 {
            quantize(&module, &stack, 4);
        } else {
            both.push((format!("{module}.weight"), stack));
        }
    }

    let write = |name: &str, tensors: &[(String, Array)], quantization: bool| {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_path("deepseek_native_v3_config.json")).unwrap(),
        )
        .unwrap();
        if quantization {
            let mut block = serde_json::json!({"group_size": GROUP, "bits": 4});
            block.as_object_mut().unwrap().extend(overrides.clone());
            config["quantization"] = block;
        }
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let entries: Vec<(&str, &Array)> = tensors
            .iter()
            .chain(both.iter())
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        Array::save_safetensors(
            dir.path().join("model.safetensors").to_str().unwrap(),
            &entries,
        );
        drain_bridge(name);
        let config = load_config(dir.path()).expect("config parses");
        let weights = load_model(dir.path(), &config).expect("weights load");
        drain_bridge(name);
        (dir, weights)
    };
    let (_packed_dir, packed_weights) = write("packed", &packed, true);
    let (_dense_dir, dense_weights) = write("dense", &dense, false);
    assert!(
        packed_weights.experts_packed(),
        "the experts were unpacked at load"
    );
    assert!(!dense_weights.experts_packed());

    let reference = load_shard(&fixture_path("deepseek_native_v3_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let run = |weights: &NativeWeights, chunks: &[i32]| {
        let mut cache = NativeCache::new_empty(weights);
        let mut start = 0;
        let mut logits = Vec::new();
        for &len in chunks {
            logits.push(forward_step(
                weights,
                &rows(&ids, start, start + len),
                &mut cache,
            ));
            start += len;
        }
        drain_bridge("forward");
        logits
    };
    let t = ids.dim(1);
    let want = run(&dense_weights, &[t]);
    let got = run(&packed_weights, &[t]);
    assert_all_pass(
        "mlx-quantized: prefill",
        &[ParityReport::compute_with_per_position(
            "logits", &got[0], &want[0], TOL,
        )],
    );
    // The same through the cache, one token at a time: decode attends in the
    // latent space through `W_UK`, which the packed `kv_b_proj` was split into.
    let mut steps = vec![3];
    steps.extend(std::iter::repeat_n(1, (t - 3) as usize));
    let want = run(&dense_weights, &steps);
    let got = run(&packed_weights, &steps);
    let reports: Vec<_> = got
        .iter()
        .zip(&want)
        .enumerate()
        .map(|(i, (g, w))| ParityReport::compute(&format!("step_{i}"), g, w, TOL))
        .collect();
    assert_all_pass("mlx-quantized: cached decode", &reports);
}

/// A trunk tensor the engine does not read (here DeepSeek-V3.2's sparse
/// attention indexer) is an error naming it, not a model that quietly runs
/// without it; the multi-token-prediction layer past the trunk is not.
#[test]
#[serial]
fn native_deepseek_refuses_trunk_tensors_it_does_not_read() {
    let source = load_shard(&fixture_path("deepseek_native_v3_weights.safetensors"));
    let extra = "model.layers.1.self_attn.indexer.wq_b.weight";
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("deepseek_native_v3_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    let mut entries: Vec<(&str, &Array)> = source.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let indexer = ref_tensor(&source, "model.norm.weight").clone();
    entries.push((extra, &indexer));
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    );
    let config = load_config(dir.path()).expect("config parses");
    let err = load_model(dir.path(), &config)
        .err()
        .expect("an unread trunk tensor is refused");
    assert!(err.contains(extra), "{err}");
    // The same checkpoint without it loads, MTP layer and all.
    entries.pop();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    );
    load_model(dir.path(), &config).expect("the fixture loads");
}

#[test]
#[serial]
fn native_deepseek_v3_matches_transformers() {
    check("v3", "deepseek_native_v3_weights.safetensors");
}

#[test]
#[serial]
fn native_deepseek_v2_greedy_matches_transformers() {
    check("v2_greedy", "deepseek_native_v2_weights.safetensors");
}

#[test]
#[serial]
fn native_deepseek_v2_group_limited_matches_transformers() {
    check(
        "v2_group_limited_greedy",
        "deepseek_native_v2_weights.safetensors",
    );
}

