//! Numerical parity for the Qwen4-Exp (Qwen3.8-Flash-Next) text tower against
//! the HuggingFace `transformers` `Qwen4ExpForCausalLM` oracle.
//!
//! The fixture is a tiny fp32 model with every feature switched on (dumped by
//! `.strategy/parity/dump_qwen4_exp_reference.py`): GDN with the sigmoid
//! output gate, gated attention with a QSA indexer whose budget is small enough
//! to drop blocks from position 5 on, three hyper-connection streams, PLE
//! n-gram embeddings on two layers with EOS tokens inside the prompt, and an
//! 8-expert top-3 MoE. Its weights are written in the released checkpoint
//! layout (`model.language_model.*`, fused experts, a sharded n-gram table,
//! int64 hash buffers, plus vision and MTP tensors that must be skipped), and
//! its `config.json` is the released wrapper, so the model is loaded the way
//! `pmetal infer` loads one: `DynamicModel::load`.
//!
//! Two reference runs are held to the same weights:
//!
//! * `logits`: one uncached forward over the 24-token prompt;
//! * `cached_logits`: the prompt through the KV + Mamba caches as a 9-token
//!   prefill, a 4-token chunk and then one token per step, which is what holds
//!   the GDN and PLE conv states, the n-gram token context and the indexer's
//!   block cache to the reference.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_bridge::compat::{Array, ops};
use pmetal_mlx::speculative::SpecCapture;
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_models::architectures::qwen3_next::Qwen3NextRoutedExpertMode;
use pmetal_models::architectures::qwen4_exp::{
    Qwen4ExpConfig, Qwen4ExpForCausalLM, Qwen4ExpLoadOptions, load_qwen4_exp_weights,
};
use pmetal_models::dispatcher::{DynamicModel, ModelArchitecture};
use serial_test::serial;

const TAP_LAYERS: [usize; 4] = [0, 1, 2, 3];

/// A checkpoint directory holding the fixture, as `pmetal infer` would see it.
fn checkpoint_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("qwen4_exp_synth_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("copy weights");
    std::fs::copy(
        fixture_path("qwen4_exp_synth_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    dir
}

/// fp32 throughout. Observed max |diff|: embeddings bit-exact, layers 0/1
/// 0.9e-6 / 1.1e-6, layers 2/3 4.7e-6 / 4.6e-6, final mix 2.8e-6, logits
/// 2.1e-6 (cached 1.6e-6), against the reference's own fp32-vs-fp64 error of
/// 9.2e-7 on the logits. The gates sit at about 5x the observation: well under
/// anything a real defect produces (a GDN q/k epsilon taken on the mean instead
/// of the sum of squares moves the logits by 6.3e-4, dropping the indexer by
/// far more). `passed()` is `abs || rel`, so the rtols are as tight as the
/// atols.
fn tolerances() -> Vec<(&'static str, Tolerance)> {
    vec![
        ("post_embed", Tolerance::new(0.0, 0.0)),
        ("layer_0_hidden", Tolerance::new(5e-6, 2e-6)),
        ("layer_1_hidden", Tolerance::new(5e-6, 2e-6)),
        ("layer_2_hidden", Tolerance::new(2.5e-5, 8e-6)),
        ("layer_3_hidden", Tolerance::new(2.5e-5, 8e-6)),
        ("final_hidden", Tolerance::new(1.5e-5, 8e-6)),
        ("logits", Tolerance::new(1e-5, 8e-6)),
        ("cached_logits", Tolerance::new(1e-5, 8e-6)),
    ]
}

fn compare(name: &str, rust: &Array, reference: &Array) -> ParityReport {
    let tol = tolerances()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t)
        .expect("every compared tensor has a tolerance");
    ParityReport::compute_with_per_position(name, rust, reference, tol)
}

fn assert_all_pass(label: &str, reports: &[ParityReport]) {
    println!("\n== Qwen4-Exp {label} ==");
    print_report_table(reports);
    let failures: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(
        failures.is_empty(),
        "{label}: parity failed at {failures:?}"
    );
}

fn assert_argmax(logits: &Array, reference: &Array) {
    let rust = argmax_last_axis(logits);
    let expected: Vec<i32> = to_f32_vec_eval(reference)
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(rust, expected, "argmax differs from the reference");
}

/// The cached run: a 9-token prefill, a 4-token chunk, then one token a step.
fn cached_logits(model: &mut DynamicModel, input_ids: &Array) -> Array {
    let (batch, seq) = (input_ids.dim(0), input_ids.dim(1));
    let mut kv = model.create_cache(64);
    let mut mamba = model.create_mamba_cache();
    let mut segments = vec![(0, 9), (9, 13)];
    segments.extend((13..seq).map(|t| (t, t + 1)));
    let mut outputs = Vec::new();
    for (start, stop) in segments {
        let chunk = input_ids.slice(&[0, start], &[batch, stop]);
        let logits = model
            .forward_with_hybrid_cache(&chunk, None, Some(&mut kv), mamba.as_mut())
            .expect("cached forward");
        pmetal_bridge::check_last_error().expect("no bridge error in the cached forward");
        outputs.push(logits);
    }
    let refs: Vec<&Array> = outputs.iter().collect();
    ops::concatenate_axis(&refs, 1)
}

#[test]
#[serial]
fn qwen4_exp_matches_transformers_full_and_cached() {
    let reference = load_shard(&fixture_path("qwen4_exp_synth_reference.safetensors"));
    let input_ids = ref_tensor(&reference, "input_ids").clone();
    let dir = checkpoint_dir();

    assert_eq!(
        ModelArchitecture::detect(dir.path()).expect("detects"),
        ModelArchitecture::Qwen4Exp
    );
    let mut model = DynamicModel::load(dir.path()).expect("production load");
    assert_eq!(model.architecture(), ModelArchitecture::Qwen4Exp);

    // Uncached forward with per-layer taps.
    let DynamicModel::Qwen4Exp(inner) = &mut model else {
        unreachable!("detected as Qwen4Exp");
    };
    let mut capture = SpecCapture::with_layers_and_embedding(TAP_LAYERS.to_vec(), true);
    let final_hidden = inner
        .model
        .forward_with_cache_and_capture(&input_ids, None, None, None, Some(&mut capture))
        .expect("uncached forward");
    pmetal_bridge::check_last_error().expect("no bridge error in the uncached forward");
    let logits = model.forward(&input_ids, None).expect("logits");

    let mut reports = vec![compare(
        "post_embed",
        capture.embedding.as_ref().expect("embedding tap"),
        ref_tensor(&reference, "post_embed"),
    )];
    for layer in TAP_LAYERS {
        let name = format!("layer_{layer}_hidden");
        reports.push(compare(
            &name,
            &capture.hidden_states[&layer],
            ref_tensor(&reference, &name),
        ));
    }
    reports.push(compare(
        "final_hidden",
        &final_hidden,
        ref_tensor(&reference, "final_hidden"),
    ));
    reports.push(compare("logits", &logits, ref_tensor(&reference, "logits")));
    assert_all_pass("uncached", &reports);
    assert_argmax(&logits, ref_tensor(&reference, "argmax_tokens"));

    // The same prompt through the caches.
    let cached = cached_logits(&mut model, &input_ids);
    let reports = vec![compare(
        "cached_logits",
        &cached,
        ref_tensor(&reference, "cached_logits"),
    )];
    assert_all_pass("cached", &reports);
    assert_argmax(&cached, ref_tensor(&reference, "argmax_tokens"));
}

/// Serving the n-gram rows from the checkpoint file must give exactly what the
/// resident table gives: the same bytes, looked up by the same row ids.
#[test]
#[serial]
fn qwen4_exp_disk_served_ngram_rows_match_resident() {
    let reference = load_shard(&fixture_path("qwen4_exp_synth_reference.safetensors"));
    let input_ids = ref_tensor(&reference, "input_ids").clone();
    let dir = checkpoint_dir();
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
            .unwrap();
    let config =
        Qwen4ExpConfig::from_json(&config_json["text_config"].to_string()).expect("config");

    let mut logits = Vec::new();
    for on_disk in [false, true] {
        let mut model = Qwen4ExpForCausalLM::new_for_loading(
            config.clone(),
            Qwen3NextRoutedExpertMode::Resident,
        )
        .expect("model");
        load_qwen4_exp_weights(
            &mut model,
            dir.path(),
            Qwen4ExpLoadOptions {
                ngram_rows_on_disk: on_disk,
                ..Default::default()
            },
        )
        .expect("load");
        let ple = model.model.layers[0].ple.as_ref().expect("layer 0 has PLE");
        assert_eq!(
            ple.ple_embedding.ngram_embedding.disk.is_some(),
            on_disk,
            "rows served from where they were asked to be"
        );
        logits.push(model.forward(&input_ids, None).expect("forward"));
    }
    assert_eq!(
        to_f32_vec_eval(&logits[0]),
        to_f32_vec_eval(&logits[1]),
        "disk-served n-gram rows changed the logits"
    );
}

/// The fixture with its routed experts in the NVIDIA ModelOpt NVFP4 layout the
/// `nvidia/Qwen3.8-Flash-Next-NVFP4` release ships: one module per expert, the
/// e2m1 bytes, an e4m3 scale per 16 values, and a per-expert `weight_scale_2`.
fn nvfp4_checkpoint_dir() -> tempfile::TempDir {
    use pmetal_bridge::QuantizedMode;

    let dense = load_shard(&fixture_path("qwen4_exp_synth_weights.safetensors"));
    let mut out: Vec<(String, Array)> = Vec::new();
    let scalar = |v: f32| Array::from_f32_slice(&[v], &[]);
    for (key, value) in &dense {
        let Some(prefix) = key
            .strip_suffix(".gate_up_proj")
            .or_else(|| key.strip_suffix(".down_proj"))
            .filter(|p| p.ends_with(".mlp.experts"))
        else {
            out.push((key.clone(), value.clone()));
            continue;
        };
        let (experts, rows, cols) = (value.dim(0), value.dim(1), value.dim(2));
        let projections: Vec<(&str, i32, i32)> = if key.ends_with(".gate_up_proj") {
            vec![("gate_proj", 0, rows / 2), ("up_proj", rows / 2, rows)]
        } else {
            vec![("down_proj", 0, rows)]
        };
        for e in 0..experts {
            for &(name, start, stop) in &projections {
                let w = value
                    .slice(&[e, start, 0], &[e + 1, stop, cols])
                    .reshape(&[stop - start, cols]);
                let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
                let module = format!("{prefix}.{e}.{name}");
                out.push((
                    format!("{module}.weight"),
                    packed.view(pmetal_bridge::dtype::U8),
                ));
                out.push((format!("{module}.weight_scale"), scales));
                out.push((
                    format!("{module}.weight_scale_2"),
                    // Distinct per expert, and powers of two so the dense
                    // unpacking is exact and the comparison isolates the kernel.
                    scalar(2f32.powi(e % 4 - 2)),
                ));
                out.push((format!("{module}.input_scale"), scalar(1.0)));
            }
        }
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let entries: Vec<(&str, &Array)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    );
    std::fs::copy(
        fixture_path("qwen4_exp_synth_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    dir
}

/// NVFP4 experts stay packed and run on MLX's quantized gather kernels, with
/// each expert's tensor scale applied to its product. Against the same bytes
/// unpacked to dense, uncached and cached, the logits agree to fp32
/// accumulation order.
#[test]
#[serial]
fn qwen4_exp_nvfp4_experts_run_packed() {
    let reference = load_shard(&fixture_path("qwen4_exp_synth_reference.safetensors"));
    let input_ids = ref_tensor(&reference, "input_ids").clone();
    let dir = nvfp4_checkpoint_dir();

    let mut runs = Vec::new();
    for unpack in [true, false] {
        let mut model = DynamicModel::load(dir.path()).expect("loads");
        if unpack {
            let DynamicModel::Qwen4Exp(inner) = &mut model else {
                unreachable!()
            };
            let config = inner.config.clone();
            *inner =
                Qwen4ExpForCausalLM::new_for_loading(config, Qwen3NextRoutedExpertMode::Resident)
                    .expect("model");
            load_qwen4_exp_weights(
                inner,
                dir.path(),
                Qwen4ExpLoadOptions {
                    unpack_quantized_experts: true,
                    ..Default::default()
                },
            )
            .expect("dense load");
        }
        let DynamicModel::Qwen4Exp(inner) = &model else {
            unreachable!()
        };
        assert_eq!(
            inner.model.layers[0].mlp.packed_experts.is_some(),
            !unpack,
            "experts packed only when asked"
        );
        let logits = model.forward(&input_ids, None).expect("forward");
        let cached = cached_logits(&mut model, &input_ids);
        pmetal_bridge::check_last_error().expect("no bridge error");
        runs.push((logits, cached));
    }
    // Observed 6.6e-7 both ways; swapping a gate and up projection moves the
    // logits by 0.79.
    let tol = Tolerance::new(5e-6, 4e-6);
    let reports = vec![
        ParityReport::compute("nvfp4_packed_vs_dense", &runs[1].0, &runs[0].0, tol),
        ParityReport::compute("nvfp4_packed_vs_dense_cached", &runs[1].1, &runs[0].1, tol),
    ];
    assert_all_pass("NVFP4 packed experts", &reports);
    assert_eq!(argmax_last_axis(&runs[1].0), argmax_last_axis(&runs[0].0));
}

/// Two prompts in one batch, through the caches, give each prompt's own
/// logits: the n-gram context, PLE conv and indexer states are kept per row.
#[test]
#[serial]
fn qwen4_exp_batch_rows_are_independent() {
    let reference = load_shard(&fixture_path("qwen4_exp_synth_reference.safetensors"));
    let first = ref_tensor(&reference, "input_ids").clone();
    let ids: Vec<i32> = to_f32_vec_eval(&first).iter().map(|&t| t as i32).collect();
    let reversed: Vec<i32> = ids.iter().rev().copied().collect();
    let second = Array::from_i32_slice_shaped(&reversed, &[1, ids.len() as i32]);
    let both =
        Array::from_i32_slice_shaped(&[ids.clone(), reversed].concat(), &[2, ids.len() as i32]);

    let dir = checkpoint_dir();
    let mut model = DynamicModel::load(dir.path()).expect("loads");
    let alone = [
        cached_logits(&mut model, &first),
        cached_logits(&mut model, &second),
    ];
    let batched = cached_logits(&mut model, &both);
    for (row, logits) in alone.iter().enumerate() {
        let slice = batched.slice(
            &[row as i32, 0, 0],
            &[row as i32 + 1, batched.dim(1), batched.dim(2)],
        );
        let report = ParityReport::compute(
            &format!("batch_row_{row}"),
            &slice,
            logits,
            Tolerance::new(1e-5, 1e-5),
        );
        assert_all_pass("batched cached decode", &[report]);
    }
}
