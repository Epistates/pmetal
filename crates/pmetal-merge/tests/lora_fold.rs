//! Folding LoRA adapters into a base model: W' = W + s * B A with s the
//! adapter's scale (alpha / r, or alpha / sqrt(r) under rsLoRA), DoRA's
//! magnitude renormalisation, both adapter layouts, and loud failure when the
//! adapter does not fit the base.

mod common;

use common::{assert_close, write_file};
use pmetal_merge::{AccurateMergeConfig, streaming_lora_merge};
use safetensors::Dtype;
use std::path::Path;

const Q: &str = "model.layers.0.self_attn.q_proj";

/// W = [[1, 2, 3], [4, 5, 6]]
fn write_base(dir: &Path) {
    write_file(
        dir,
        "model.safetensors",
        &[
            (
                &format!("{Q}.weight"),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![2, 3],
            ),
            ("model.norm.weight", vec![1.0, 1.0], vec![2]),
        ],
        Dtype::F32,
    );
    std::fs::write(dir.join("config.json"), "{}").unwrap();
}

/// A = [[1, 0, 0], [0, 1, 0]] (r = 2), B = [[1, 0], [0, 2]], so B A =
/// [[1, 0, 0], [0, 2, 0]].
const A: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
const B: [f32; 4] = [1.0, 0.0, 0.0, 2.0];

fn write_peft_adapter(
    dir: &Path,
    config: serde_json::Value,
    extra: &[(&str, Vec<f32>, Vec<usize>)],
) {
    let a_key = format!("base_model.model.{Q}.lora_A.weight");
    let b_key = format!("base_model.model.{Q}.lora_B.weight");
    let mut tensors: Vec<(&str, Vec<f32>, Vec<usize>)> = vec![
        (&a_key, A.to_vec(), vec![2, 3]),
        (&b_key, B.to_vec(), vec![2, 2]),
    ];
    tensors.extend(extra.iter().cloned());
    write_file(dir, "adapter_model.safetensors", &tensors, Dtype::F32);
    std::fs::write(dir.join("adapter_config.json"), config.to_string()).unwrap();
}

fn fold(base: &Path, adapter: &Path, out: &Path) -> pmetal_merge::Result<()> {
    streaming_lora_merge(&AccurateMergeConfig::new(base, adapter, out)).map(|_| ())
}

fn merged_q(out: &Path) -> Vec<f32> {
    common::read_tensor_file(&out.join("model.safetensors"), &format!("{Q}.weight"))
}

#[test]
fn folds_with_alpha_over_rank() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_peft_adapter(
        &adapter,
        serde_json::json!({"r": 2, "lora_alpha": 4.0}),
        &[],
    );
    fold(&base, &adapter, &out).unwrap();
    // scale 4/2 = 2: W + 2 * [[1,0,0],[0,2,0]]
    assert_close(&merged_q(&out), &[3.0, 2.0, 3.0, 4.0, 9.0, 6.0], 1e-6);
}

#[test]
fn rslora_scales_by_alpha_over_sqrt_rank() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_peft_adapter(
        &adapter,
        serde_json::json!({"r": 2, "lora_alpha": 4.0, "use_rslora": true}),
        &[],
    );
    fold(&base, &adapter, &out).unwrap();
    let s = 4.0_f32 / 2.0_f32.sqrt();
    assert_close(
        &merged_q(&out),
        &[1.0 + s, 2.0, 3.0, 4.0, 5.0 + 2.0 * s, 6.0],
        1e-5,
    );
}

#[test]
fn alpha_pattern_overrides_alpha_for_matching_modules() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_peft_adapter(
        &adapter,
        serde_json::json!({"r": 2, "lora_alpha": 4.0, "alpha_pattern": {"q_proj": 8.0}}),
        &[],
    );
    fold(&base, &adapter, &out).unwrap();
    // scale 8/2 = 4
    assert_close(&merged_q(&out), &[5.0, 2.0, 3.0, 4.0, 13.0, 6.0], 1e-6);
}

/// DoRA (Liu et al. 2024): W' = m * (W + s B A) / ||W + s B A||, the norm
/// taken over each output row.
#[test]
fn dora_renormalizes_each_row_to_the_magnitude() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    let key = format!("base_model.model.{Q}.lora_magnitude_vector");
    write_peft_adapter(
        &adapter,
        serde_json::json!({"r": 2, "lora_alpha": 4.0, "use_dora": true}),
        &[(&key, vec![10.0, 1.0], vec![2])],
    );
    fold(&base, &adapter, &out).unwrap();

    let combined: [[f64; 3]; 2] = [[3.0, 2.0, 3.0], [4.0, 9.0, 6.0]];
    let magnitude = [10.0_f64, 1.0];
    let mut want = Vec::new();
    for (row, m) in combined.iter().zip(magnitude) {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        want.extend(row.iter().map(|v| (v * m / norm) as f32));
    }
    assert_close(&merged_q(&out), &want, 1e-5);
}

/// The layout `pmetal train` writes: lora_weights.safetensors, keys
/// `<module>.lora_a` / `.lora_b` / `.lora_magnitude`, and an adapter_config
/// with `alpha` and `use_rslora`.
#[test]
fn folds_adapters_in_the_native_layout() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_file(
        &adapter,
        "lora_weights.safetensors",
        &[
            ("layers.0.self_attn.q_proj.lora_a", A.to_vec(), vec![2, 3]),
            ("layers.0.self_attn.q_proj.lora_b", B.to_vec(), vec![2, 2]),
        ],
        Dtype::F32,
    );
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"r": 2, "alpha": 4.0, "use_rslora": false, "target_modules": ["q_proj"]}"#,
    )
    .unwrap();
    fold(&base, &adapter, &out).unwrap();
    assert_close(&merged_q(&out), &[3.0, 2.0, 3.0, 4.0, 9.0, 6.0], 1e-6);
}

#[test]
fn native_dora_magnitude_is_a_column_vector() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_file(
        &adapter,
        "lora_weights.safetensors",
        &[
            ("layers.0.self_attn.q_proj.lora_a", A.to_vec(), vec![2, 3]),
            ("layers.0.self_attn.q_proj.lora_b", B.to_vec(), vec![2, 2]),
            (
                "layers.0.self_attn.q_proj.lora_magnitude",
                vec![10.0, 1.0],
                vec![2, 1],
            ),
        ],
        Dtype::F32,
    );
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"r": 2, "alpha": 4.0}"#,
    )
    .unwrap();
    fold(&base, &adapter, &out).unwrap();
    let norm0 = (22.0_f64).sqrt();
    let got = merged_q(&out);
    assert!(
        (f64::from(got[0]) - 3.0 * 10.0 / norm0).abs() < 1e-5,
        "{got:?}"
    );
}

#[test]
fn an_adapter_that_matches_nothing_is_an_error_not_a_silent_copy() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_file(
        &adapter,
        "adapter_model.safetensors",
        &[
            (
                "base_model.model.model.layers.9.mlp.up_proj.lora_A.weight",
                A.to_vec(),
                vec![2, 3],
            ),
            (
                "base_model.model.model.layers.9.mlp.up_proj.lora_B.weight",
                B.to_vec(),
                vec![2, 2],
            ),
        ],
        Dtype::F32,
    );
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"r": 2, "lora_alpha": 4.0}"#,
    )
    .unwrap();
    let err = fold(&base, &adapter, &out).unwrap_err().to_string();
    assert!(err.contains("layers.9.mlp.up_proj"), "{err}");
}

#[test]
fn a_half_missing_pair_is_an_error() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    let a_key = format!("base_model.model.{Q}.lora_A.weight");
    write_file(
        &adapter,
        "adapter_model.safetensors",
        &[(&a_key, A.to_vec(), vec![2, 3])],
        Dtype::F32,
    );
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"r": 2, "lora_alpha": 4.0}"#,
    )
    .unwrap();
    assert!(fold(&base, &adapter, &out).is_err());
}

#[test]
fn the_output_is_complete_or_absent() {
    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    write_base(&base);
    write_peft_adapter(
        &adapter,
        serde_json::json!({"r": 2, "lora_alpha": 4.0}),
        &[],
    );
    fold(&base, &adapter, &out).unwrap();
    let names: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().all(|n| !n.contains("partial")),
        "scratch files left behind: {names:?}"
    );
}

/// An MLX affine-quantized base (4-bit, group size 32): the fold dequantizes,
/// adds `s B A` and quantizes again, so the dequantized output equals the
/// dense sum to within a quantization step.
#[test]
fn folds_into_a_quantized_base_by_dequantize_fold_requantize() {
    use pmetal_runtime::compat::Array;

    let (out_features, in_features, group_size, bits) = (4usize, 64usize, 32, 4);
    let packed_cols = in_features * bits as usize / 32;
    let groups = in_features / group_size as usize;
    let dense: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let (packed, scales, biases) =
        Array::from_f32_slice(&dense, &[out_features as i32, in_features as i32])
            .quantize_weights(group_size, bits);
    let words: Vec<u32> = packed.as_slice::<u32>().to_vec();
    // The f16 the checkpoint stores the scales and biases in.
    let to_f16 = |a: &Array| -> Vec<f32> {
        a.clone()
            .to_f32_vec(a.size())
            .unwrap()
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_f32())
            .collect()
    };
    let (scales_f16, biases_f16) = (to_f16(&scales), to_f16(&biases));

    let work = tempfile::tempdir().unwrap();
    let (base, adapter, out) = (
        work.path().join("b"),
        work.path().join("a"),
        work.path().join("o"),
    );
    std::fs::create_dir_all(&base).unwrap();
    let packed_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let f16_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect()
    };
    let (s_bytes, b_bytes) = (f16_bytes(&scales_f16), f16_bytes(&biases_f16));
    let views = std::collections::HashMap::from([
        (
            format!("{Q}.weight"),
            safetensors::tensor::TensorView::new(
                Dtype::U32,
                vec![out_features, packed_cols],
                &packed_bytes,
            )
            .unwrap(),
        ),
        (
            format!("{Q}.scales"),
            safetensors::tensor::TensorView::new(Dtype::F16, vec![out_features, groups], &s_bytes)
                .unwrap(),
        ),
        (
            format!("{Q}.biases"),
            safetensors::tensor::TensorView::new(Dtype::F16, vec![out_features, groups], &b_bytes)
                .unwrap(),
        ),
    ]);
    std::fs::write(
        base.join("model.safetensors"),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
    std::fs::write(
        base.join("config.json"),
        r#"{"quantization": {"group_size": 32, "bits": 4}}"#,
    )
    .unwrap();

    // r = 2 adapter at scale 1 (alpha = r).
    let a: Vec<f32> = (0..2 * in_features)
        .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
        .collect();
    let b = vec![0.5, -0.25, 0.1, 0.2, -0.3, 0.4, 0.0, 0.6];
    let a_key = format!("base_model.model.{Q}.lora_A.weight");
    let b_key = format!("base_model.model.{Q}.lora_B.weight");
    write_file(
        &adapter,
        "adapter_model.safetensors",
        &[
            (&a_key, a.clone(), vec![2, in_features]),
            (&b_key, b.clone(), vec![out_features, 2]),
        ],
        Dtype::F32,
    );
    std::fs::write(
        adapter.join("adapter_config.json"),
        r#"{"r": 2, "lora_alpha": 2.0}"#,
    )
    .unwrap();

    fold(&base, &adapter, &out).unwrap();

    let dequantize = |words: &[u32], s: &[f32], bi: &[f32]| -> Vec<f32> {
        Array::from_u32_slice(words, &[out_features as i32, packed_cols as i32])
            .dequantize(
                &Array::from_f32_slice(s, &[out_features as i32, groups as i32]),
                &Array::from_f32_slice(bi, &[out_features as i32, groups as i32]),
                group_size,
                bits,
            )
            .to_f32_vec(out_features * in_features)
            .unwrap()
    };
    // Reference: the stored base, dequantized, plus B A in f32.
    let stored = dequantize(&words, &scales_f16, &biases_f16);
    let mut want = stored.clone();
    for o in 0..out_features {
        for i in 0..in_features {
            let delta: f32 = (0..2).map(|r| b[o * 2 + r] * a[r * in_features + i]).sum();
            want[o * in_features + i] += delta;
        }
    }

    let merged = std::fs::read(out.join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&merged).unwrap();
    let words_out: Vec<u32> = st
        .tensor(&format!("{Q}.weight"))
        .unwrap()
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let read = |name: &str| common::read_tensor_file(&out.join("model.safetensors"), name);
    let (s_out, b_out) = (read(&format!("{Q}.scales")), read(&format!("{Q}.biases")));
    let got = dequantize(&words_out, &s_out, &b_out);
    pmetal_runtime::check_unobserved_error().unwrap();

    // Within one quantization step (MLX shifts each group's grid so that its
    // larger edge is exactly representable, which can cost up to a step at
    // the other edge) plus the f16 rounding of the scales.
    let step = s_out.iter().cloned().fold(0.0_f32, f32::max);
    assert_ne!(got, stored, "the adapter changed nothing");
    assert_close(&got, &want, step + 0.01);
}
