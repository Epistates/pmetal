//! NVIDIA ModelOpt NVFP4 routed experts on the `DynamicModel` Qwen3.5 engine
//! stay packed and run on MLX's quantized gather kernels, with each expert's
//! tensor scale applied to its product, instead of unpacking to a dense copy
//! four times the size.
//!
//! The `qwen3_5_moe` parity fixture with its experts in the layout the
//! `nvidia/Qwen3.6-35B-A3B-NVFP4` release ships (one module per expert, the
//! e2m1 bytes, an e4m3 scale per 16 values, a per-expert `weight_scale_2`),
//! run packed and run unpacked to dense from the same bytes.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, argmax_last_axis};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::qwen3_next::{Qwen3NextForCausalLM, Qwen3NextRoutedExpertMode};
use pmetal_models::loader::{Qwen3NextLoadOptions, load_qwen3_next_weights_with_options};
use pmetal_runtime::QuantizedMode;
use pmetal_runtime::compat::Array;
use serial_test::serial;

fn nvfp4_checkpoint_dir() -> tempfile::TempDir {
    let dense = load_shard(&fixture_path("qwen3_5_moe_weights.safetensors"));
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
                    packed.view(pmetal_runtime::dtype::U8),
                ));
                out.push((format!("{module}.weight_scale"), scales));
                // Distinct per expert, and powers of two so the dense
                // unpacking is exact and the comparison isolates the kernel.
                out.push((
                    format!("{module}.weight_scale_2"),
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
    )
    .expect("save the weights");
    std::fs::copy(
        fixture_path("qwen3_5_moe_config.json"),
        dir.path().join("config.json"),
    )
    .expect("copy config");
    dir
}

#[test]
#[serial]
fn nvfp4_experts_stay_packed_and_match_their_dense_unpacking() {
    let reference = load_shard(&fixture_path("qwen3_5_moe_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let dir = nvfp4_checkpoint_dir();

    let mut packed = DynamicModel::load(dir.path()).expect("packed load");
    let DynamicModel::Qwen3Next(inner) = &packed else {
        panic!("Qwen3Next");
    };
    let config = inner.config().clone();
    assert!(
        inner.model.layers.iter().all(|l| matches!(
            &l.mlp,
            pmetal_models::architectures::qwen3_next::Qwen3NextFeedForward::MoE(b)
                if b.packed_experts.is_some()
        )),
        "every sparse layer keeps its experts packed"
    );

    let mut dense = Qwen3NextForCausalLM::new_with_routed_expert_mode(
        config.clone(),
        Qwen3NextRoutedExpertMode::Resident,
    )
    .expect("model");
    load_qwen3_next_weights_with_options(
        &mut dense,
        dir.path(),
        &config,
        Qwen3NextLoadOptions {
            unpack_quantized_experts: true,
            ..Default::default()
        },
    )
    .expect("dense load");

    let want = dense.forward(&ids, None).expect("dense forward");
    let got = packed.forward(&ids, None).expect("packed forward");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    // Swapping a gate and an up projection, or dropping a tensor scale, moves
    // the logits by O(1).
    let report = ParityReport::compute(
        "nvfp4_packed_vs_dense",
        &got,
        &want,
        Tolerance::new(5e-5, 5e-5),
    );
    assert!(report.passed(), "max |diff| {:.3e}", report.max_abs_diff);
    assert_eq!(argmax_last_axis(&got), argmax_last_axis(&want));
}
