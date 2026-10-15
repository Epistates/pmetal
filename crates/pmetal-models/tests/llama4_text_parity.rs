//! Llama 4 text against the authoritative Hugging Face `transformers`
//! `Llama4ForCausalLM`, on both of pmetal's engines, over a whole prompt and
//! through the caches one token at a time.
//!
//! The fixture (transformers 5.19, torch 2.14.1, seed 7) is a whole
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
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis, print_report_table};
use pmetal_runtime::compat::{Array, Dtype, ops::slice_axis};
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

/// The `llama4_<variant>_*` fixture as a checkpoint directory.
fn checkpoint(variant: &str) -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path(&format!("llama4_{variant}_config.json")),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path(&format!("llama4_{variant}_weights.safetensors")),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    Fixture {
        dir,
        reference: load_shard(&fixture_path(&format!(
            "llama4_{variant}_reference.safetensors"
        ))),
    }
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn drain_runtime(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
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
/// LoRA and `serve` run: an uncached forward, the same with explicit
/// positions, and a cached decode.
fn check_dynamic(variant: &str) {
    use pmetal_models::DynamicModel;

    let fx = checkpoint(variant);
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let logits = model.forward(&ids, None).expect("forward");
    drain_runtime("dynamic forward");
    assert_all_pass(
        &format!("{variant}: dynamic forward"),
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
    drain_runtime("dynamic forward with positions");
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
        drain_runtime("dynamic decode step");
    }
    assert_all_pass(
        &format!("{variant}: dynamic cached decode"),
        &step_reports(&steps, &want),
    );
}

/// The native engine (`pmetal_runtime::llama4_native`): a prefill, then a
/// cached decode through its compiled single-token graph and through the
/// per-op path, which must agree with each other and with transformers.
fn check_native(variant: &str) {
    use pmetal_runtime::llama4_native::{NativeCache, forward_step, load_config, load_model};

    let fx = checkpoint(variant);
    let config = load_config(fx.path()).expect("native config parses");
    let weights = load_model(fx.path(), &config).expect("native weights load");
    drain_runtime("native load");
    let ids = fx.tensor("input_ids");
    let t = ids.dim(1);
    let want = fx.tensor("logits");

    let mut cache = NativeCache::new_empty(&weights);
    let logits = forward_step(&weights, &ids, &mut cache);
    drain_runtime("native prefill");
    assert_all_pass(
        &format!("{variant}: native prefill"),
        &[ParityReport::compute_with_per_position(
            "logits", &logits, &want, TOL,
        )],
    );

    let decode = |compiled: bool| {
        pmetal_runtime::decode::set_compiled_decode(compiled);
        let mut cache = NativeCache::new_empty(&weights);
        let mut steps = vec![forward_step(&weights, &rows(&ids, 0, PREFILL), &mut cache)];
        for pos in PREFILL..t {
            steps.push(forward_step(
                &weights,
                &rows(&ids, pos, pos + 1),
                &mut cache,
            ));
            drain_runtime("native decode step");
        }
        pmetal_runtime::decode::set_compiled_decode(true);
        steps
    };
    let compiled = decode(true);
    let per_op = decode(false);
    assert_all_pass(
        &format!("{variant}: native cached decode"),
        &step_reports(&compiled, &want),
    );
    let graph_vs_ops = compiled
        .iter()
        .zip(&per_op)
        .map(|(a, b)| max_abs_diff(a, b))
        .fold(0f32, f32::max);
    println!("{variant}: compiled decode vs per-op decode: max |diff| {graph_vs_ops:e}");
    assert!(
        graph_vs_ops <= 1e-5,
        "{variant}: the compiled decode graph drifted from the per-op path by {graph_vs_ops:e}"
    );

    // The prompt prefilled in passes through the last-row forward, as
    // generation does for a long prompt: the logits of the final position and
    // no other row's, with passes that start inside and past an attention
    // chunk.
    let prompt: Vec<u32> = {
        ids.eval();
        ids.as_slice::<i32>().iter().map(|&t| t as u32).collect()
    };
    for chunk in [4, 5, usize::MAX] {
        let mut cache = NativeCache::new_empty(&weights);
        let last = pmetal_runtime::decode::prefill_last_logits(
            &weights,
            &mut cache,
            &prompt,
            chunk,
            pmetal_runtime::llama4_native::forward_step_last,
        );
        drain_runtime("last-row prefill");
        assert_eq!(last.dim(0), 1);
        assert_eq!(
            last.dim(1),
            want.dim(2),
            "one row of logits, not one per token"
        );
        assert_all_pass(
            &format!("{variant}: native last-row prefill in passes of {chunk}"),
            &[ParityReport::compute(
                "last_logits",
                &last.expand_dims(1),
                &rows(&want, t - 1, t),
                TOL,
            )],
        );
    }
}

/// Dense layers only: rotation, QK-norm and temperature at the cache's
/// offset. The `DynamicModel` path rotated every forward from position 0
/// and kept no cache, so a cached decode attended to nothing but the new
/// token.
#[test]
#[serial]
fn dynamic_llama4_matches_transformers() {
    check_dynamic("text");
}

#[test]
#[serial]
fn native_llama4_matches_transformers() {
    check_native("text");
}

/// Layers 1 and 3 are MoE with the checkpoint's fused experts
/// (`experts.gate_up_proj [E, H, 2I]`, `experts.down_proj [E, I, H]`, applied
/// `x @ W`), expert width 48 against hidden 32 so a transposed tensor cannot
/// pass, and a top-1 router whose sigmoid gate scales the expert's input.
#[test]
#[serial]
fn dynamic_llama4_moe_matches_transformers() {
    check_dynamic("moe");
}

#[test]
#[serial]
fn native_llama4_moe_matches_transformers() {
    check_native("moe");
}

/// An MLX-style 4-bit quantization of the multimodal fixture (group 32, the
/// router at 8 bits, the experts split into MLX's stacked `[E, out, in]`
/// projections, the tied embedding packed so the head runs `as_linear` on
/// it), and the same weights dequantized to dense in the same layout. The
/// packed model has to give the dense one's logits over a prefill and a cached
/// decode (packed attention decodes per op, the dense twin through the
/// compiled graph).
///
/// What does not divide into groups (the 48-wide shared expert's
/// `down_proj` and the experts' own) stays dense in both, as a
/// mixed-precision checkpoint leaves it.
#[test]
#[serial]
fn native_llama4_runs_an_mlx_quantized_checkpoint() {
    use pmetal_runtime::llama4_native::{NativeCache, forward_step, load_config, load_model};

    const GROUP: i32 = 32;
    let source = load_shard(&fixture_path("llama4_native_weights.safetensors"));
    let mut packed: Vec<(String, Array)> = Vec::new();
    let mut dense: Vec<(String, Array)> = Vec::new();
    let mut both: Vec<(String, Array)> = Vec::new();
    let mut overrides = serde_json::Map::new();
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
    let mut keys: Vec<&String> = source.keys().collect();
    keys.sort();
    for key in keys {
        let w = &source[key];
        if let Some(prefix) = key.strip_suffix(".experts.gate_up_proj") {
            // `[E, H, 2I]`, gate then up, applied `x @ W`: MLX's layout is
            // one `[E, I, H]` projection each.
            let inter = w.dim(2) / 2;
            let halves = w.split(&[inter], 2);
            for (name, half) in ["gate_proj", "up_proj"].iter().zip(&halves) {
                quantize(
                    &format!("{prefix}.experts.{name}"),
                    &half.swap_axes(1, 2),
                    4,
                );
            }
        } else if let Some(prefix) = key.strip_suffix(".experts.down_proj") {
            both.push((
                format!("{prefix}.experts.down_proj.weight"),
                w.swap_axes(1, 2),
            ));
        } else if let (Some(module), true) = (
            key.strip_suffix(".weight"),
            w.ndim() == 2 && w.dim(1) % GROUP == 0 && !key.contains("norm"),
        ) {
            let bits = if module.ends_with(".router") { 8 } else { 4 };
            quantize(module, w, bits);
        } else {
            both.push((key.clone(), w.clone()));
        }
    }

    let write = |name: &str, tensors: &[(String, Array)], quantization: bool| {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixture_path("llama4_native_config.json")).unwrap(),
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
        )
        .expect("save the weights");
        drain_runtime(name);
        let config = load_config(dir.path()).expect("config parses");
        let weights = load_model(dir.path(), &config).expect("weights load");
        drain_runtime(name);
        (dir, weights)
    };
    let (_packed_dir, packed_weights) = write("packed", &packed, true);
    let (_dense_dir, dense_weights) = write("dense", &dense, false);
    assert!(
        packed_weights.is_packed(),
        "the weights were unpacked at load"
    );
    assert!(!dense_weights.is_packed());

    let reference = load_shard(&fixture_path("llama4_native_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let t = ids.dim(1);
    let run = |weights: &pmetal_runtime::llama4_native::NativeWeights, chunks: &[i32]| {
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
        drain_runtime("forward");
        logits
    };
    let mut steps = vec![PREFILL];
    steps.extend(std::iter::repeat_n(1, (t - PREFILL) as usize));
    for (title, chunks) in [("prefill", vec![t]), ("cached decode", steps)] {
        let want = run(&dense_weights, &chunks);
        let got = run(&packed_weights, &chunks);
        let reports: Vec<_> = got
            .iter()
            .zip(&want)
            .enumerate()
            .map(|(i, (g, w))| ParityReport::compute(&format!("step_{i}"), g, w, TOL))
            .collect();
        assert_all_pass(&format!("mlx-quantized: {title}"), &reports);
    }
}

/// The released multimodal layout: `model_type: llama4` with the text model
/// under `text_config` and `tie_word_embeddings` at the top level only (here
/// true, so there is no `lm_head`), tensors under `language_model.model.*`
/// beside vision tower and projector tensors, a dense layer between the MoE
/// ones, and the router taking its top 2: gated expert outputs summed, which
/// every release (top-1) never exercises but transformers' router defines.
#[test]
#[serial]
fn native_llama4_reads_the_released_multimodal_layout() {
    check_native("native");
}

/// The same checkpoint through the shared path: the fused `experts` tensors,
/// `router.weight`, `shared_expert.*` and the dense layer's MLP are its own
/// parameters under `feed_forward`, and the vision tower and projector tensors
/// are the only ones it leaves unread.
#[test]
#[serial]
fn dynamic_llama4_reads_the_released_multimodal_layout() {
    check_dynamic("native");
}

/// `attention_chunk_size` 6 on the 20-token prompt: the RoPE layers attend
/// within chunks `0..6, 6..12, …` (transformers' chunked causal mask), the
/// NoPE layer to the whole prefix. The prefill of 7 already crosses a chunk
/// boundary, and the decode steps run through three more.
#[test]
#[serial]
fn dynamic_llama4_chunked_attention_matches_transformers() {
    check_dynamic("chunked");
}

#[test]
#[serial]
fn native_llama4_chunked_attention_matches_transformers() {
    check_native("chunked");
}
