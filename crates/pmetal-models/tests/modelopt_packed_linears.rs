//! The projections of an NVIDIA ModelOpt NVFP4 checkpoint stay packed on the
//! `DynamicModel`'s `Linear`s.
//!
//! They used to be unpacked to a dense copy as the checkpoint was read, four
//! times the size in bf16 (Qwen3.8-27B-NVFP4 is 15 GB on MLX's kernel and 54 GB
//! dense). Now the words, the e4m3 block scales and the per-tensor factor go
//! on the layer, whose forward runs the quantized matmul and multiplies the
//! product by the factor. These tests build a ModelOpt checkpoint of a tiny
//! model from its own weights and hold the packed model to the same bytes
//! unpacked to dense.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis};
use pmetal_models::DynamicModel;
use pmetal_models::loader::{Qwen3NextLoadOptions, load_qwen3_next_weights_with_options};
use pmetal_runtime::QuantizedMode;
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt, VisitLinears, random};
use serial_test::serial;

const LLAMA: &str = r#"{
    "model_type": "llama", "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
    "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

fn scalar(v: f32) -> Array {
    Array::from_f32_slice(&[v], &[])
}

/// ModelOpt's NVFP4 tensors for the `[out, in]` weight `w`: the e2m1 bytes, an
/// e4m3 scale per 16, a per-tensor factor `scale_2` (a power of two, so the
/// dense unpacking is exact and the comparison isolates the kernel).
fn modelopt(out: &mut Vec<(String, Array)>, module: &str, w: &Array, scale_2: f32) {
    let (packed, scales) = w.quantize_weights_mode(16, 4, QuantizedMode::Nvfp4);
    out.push((
        format!("{module}.weight"),
        packed.view(pmetal_runtime::dtype::U8),
    ));
    out.push((format!("{module}.weight_scale"), scales));
    out.push((format!("{module}.weight_scale_2"), scalar(scale_2)));
    out.push((format!("{module}.input_scale"), scalar(1.0)));
}

/// A tiny Llama's checkpoint with every projection but the head in ModelOpt
/// NVFP4, and its linear paths.
fn llama_checkpoint() -> (tempfile::TempDir, Vec<String>) {
    random::seed(91);
    let mut model = DynamicModel::from_config(LLAMA).expect("build");
    let mut linears = Vec::new();
    model.visit_linears_mut("", &mut |path, linear| {
        if path != "lm_head" {
            linears.push((path.to_string(), linear.weight.value.clone()));
        }
    });
    let quantized: HashMap<&str, &Array> = linears.iter().map(|(p, w)| (p.as_str(), w)).collect();
    let mut out: Vec<(String, Array)> = Vec::new();
    for (name, value) in model.flatten_params() {
        let name = name.to_string();
        match name.strip_suffix(".weight").and_then(|m| quantized.get(m)) {
            Some(w) => {
                let scale_2 = 2f32.powi((name.len() % 3) as i32 - 1);
                modelopt(&mut out, name.strip_suffix(".weight").unwrap(), w, scale_2);
            }
            None => out.push((name, value)),
        }
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let entries: Vec<(&str, &Array)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save");
    std::fs::write(dir.path().join("config.json"), LLAMA).expect("config");
    (dir, linears.into_iter().map(|(p, _)| p).collect())
}

fn ids() -> Array {
    let ids: Vec<i32> = (0..24).map(|i| (i * 7 + 3) % 256).collect();
    Array::from_slice(&ids, &[1, 24])
}

#[test]
#[serial]
fn a_modelopt_llama_keeps_its_projections_packed_and_agrees_with_the_dense_unpacking() {
    let (dir, linears) = llama_checkpoint();
    let mut packed = DynamicModel::load(dir.path()).expect("packed load");

    let mut seen = Vec::new();
    packed.visit_linears_mut("", &mut |path, linear| {
        if let Some(quant) = &linear.quant {
            assert_eq!(quant.params.mode, QuantizedMode::Nvfp4, "{path}");
            assert!(quant.tensor_scale.is_some(), "{path}");
            assert_eq!(linear.weight.value.dtype(), Dtype::Uint32, "{path}");
            seen.push(path.to_string());
        }
    });
    seen.sort();
    let mut want = linears.clone();
    want.sort();
    assert_eq!(seen, want, "every projection but the head is packed");

    // The same bytes, unpacked.
    let mut dense = DynamicModel::load(dir.path()).expect("load");
    dense.visit_linears_mut("", &mut |_, linear| linear.dequantize());

    let a = packed.forward(&ids(), None).expect("packed forward");
    let b = dense.forward(&ids(), None).expect("dense forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    let report = ParityReport::compute("modelopt_llama", &a, &b, Tolerance::new(5e-4, 5e-4));
    assert!(report.passed(), "max |diff| {:.3e}", report.max_abs_diff);
    assert_eq!(argmax_last_axis(&a), argmax_last_axis(&b));

    // And they are still packed in memory: the projections' words and scales
    // take about a fifth of the dense float32 weights.
    let bytes = |model: &DynamicModel| -> usize {
        model
            .flatten_params()
            .into_iter()
            .filter(|(k, _)| k.contains(".layers."))
            .map(|(_, v)| v.nbytes())
            .sum()
    };
    assert!(
        bytes(&packed) * 3 < bytes(&dense),
        "{} packed bytes against {} dense",
        bytes(&packed),
        bytes(&dense)
    );
}

/// The Qwen3.5 sparse fixture (experts as in the NVIDIA release), with its
/// dense projections in ModelOpt NVFP4 too: packed against the same bytes
/// unpacked by the loader's own option.
#[test]
#[serial]
fn modelopt_qwen3_5_projections_stay_packed_beside_the_experts() {
    let dense_shard = load_shard(&fixture_path("qwen3_5_moe_weights.safetensors"));
    let reference = load_shard(&fixture_path("qwen3_5_moe_reference.safetensors"));
    let input_ids = ref_tensor(&reference, "input_ids").clone();

    // Which parameters are projections: ask a model built from the fixture.
    let plain_dir = tempfile::tempdir().unwrap();
    {
        let entries: Vec<(&str, &Array)> =
            dense_shard.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(
            plain_dir.path().join("model.safetensors").to_str().unwrap(),
            &entries,
        )
        .unwrap();
        std::fs::copy(
            fixture_path("qwen3_5_moe_config.json"),
            plain_dir.path().join("config.json"),
        )
        .unwrap();
    }
    let mut probe = DynamicModel::load(plain_dir.path()).expect("dense fixture loads");
    let mut projections = Vec::new();
    probe.visit_linears_mut("", &mut |path, linear| {
        if path != "lm_head" && linear.shape().1 % 16 == 0 {
            projections.push(path.to_string());
        }
    });
    assert!(projections.len() >= 8, "{projections:?}");

    let mut out: Vec<(String, Array)> = Vec::new();
    for (key, value) in &dense_shard {
        // The fixture may spell a module with its wrapper's prefix; the model's
        // paths are the canonical ones.
        let canonical = pmetal_runtime::qwen3_native::family::canonical_checkpoint_key(key);
        let is_projection = canonical
            .strip_suffix(".weight")
            .is_some_and(|m| projections.iter().any(|p| p == m));
        if is_projection {
            modelopt(&mut out, key.strip_suffix(".weight").unwrap(), value, 0.5);
        } else if let Some(prefix) = key
            .strip_suffix(".gate_up_proj")
            .or_else(|| key.strip_suffix(".down_proj"))
            .filter(|p| p.ends_with(".mlp.experts"))
        {
            // The routed experts, one NVFP4 module each, as the release ships.
            let (experts, rows, cols) = (value.dim(0), value.dim(1), value.dim(2));
            let parts: Vec<(&str, i32, i32)> = if key.ends_with(".gate_up_proj") {
                vec![("gate_proj", 0, rows / 2), ("up_proj", rows / 2, rows)]
            } else {
                vec![("down_proj", 0, rows)]
            };
            for e in 0..experts {
                for &(name, start, stop) in &parts {
                    let w = value
                        .slice(&[e, start, 0], &[e + 1, stop, cols])
                        .reshape(&[stop - start, cols]);
                    modelopt(
                        &mut out,
                        &format!("{prefix}.{e}.{name}"),
                        &w,
                        2f32.powi(e % 4 - 2),
                    );
                }
            }
        } else {
            out.push((key.clone(), value.clone()));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let entries: Vec<(&str, &Array)> = out.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .unwrap();
    std::fs::copy(
        fixture_path("qwen3_5_moe_config.json"),
        dir.path().join("config.json"),
    )
    .unwrap();

    let mut packed = DynamicModel::load(dir.path()).expect("packed load");
    let mut packed_count = 0;
    packed.visit_linears_mut("", &mut |_, linear| {
        packed_count += usize::from(linear.quant.is_some());
    });
    assert_eq!(
        packed_count,
        projections.len(),
        "every projection is packed"
    );

    let DynamicModel::Qwen3Next(inner) = &packed else {
        panic!("Qwen3Next");
    };
    let config = inner.config().clone();
    let mut unpacked = pmetal_models::architectures::qwen3_next::Qwen3NextForCausalLM::new_with_routed_expert_mode(
        config.clone(),
        pmetal_models::architectures::qwen3_next::Qwen3NextRoutedExpertMode::Resident,
    )
    .expect("model");
    load_qwen3_next_weights_with_options(
        &mut unpacked,
        dir.path(),
        &config,
        Qwen3NextLoadOptions {
            unpack_quantized_experts: true,
            ..Default::default()
        },
    )
    .expect("unpacked load");
    let mut stays_dense = true;
    unpacked.visit_linears_mut("", &mut |_, linear| stays_dense &= linear.quant.is_none());
    assert!(stays_dense, "the unpacking option unpacks projections too");

    let got = packed.forward(&input_ids, None).expect("packed forward");
    let want = unpacked
        .forward(&input_ids, None)
        .expect("unpacked forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    let report = ParityReport::compute("modelopt_qwen3_5", &got, &want, Tolerance::new(5e-4, 5e-4));
    assert!(report.passed(), "max |diff| {:.3e}", report.max_abs_diff);
    assert_eq!(argmax_last_axis(&got), argmax_last_axis(&want));
}
