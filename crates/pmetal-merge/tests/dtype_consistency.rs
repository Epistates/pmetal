//! Integration tests for cross-model dtype consistency at merge time.
//!
//! These tests verify the contract from Phase 1.2 of the SOTA-readiness work:
//! a per-tensor dtype mismatch between input models surfaces as
//! [`pmetal_merge::MergeError::DtypeMismatch`] unless `allow_mixed_dtype` is
//! `true`, in which case all sources are upcast to f32 in-memory and merging
//! proceeds.

use pmetal_merge::{
    MergeBuilder, MergeConfig, MergeError, MergeMethodConfig, ModelConfig, run_merge,
};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

/// Write a single-tensor safetensors file at `dir/model.safetensors`. The
/// caller chooses the on-disk dtype; `values` is interpreted as f32 source
/// data and packed accordingly.
fn write_one_tensor_model(dir: &Path, name: &str, values: &[f32], dtype: Dtype) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes: Vec<u8> = match dtype {
        Dtype::F32 => values.iter().flat_map(|f| f.to_le_bytes()).collect(),
        Dtype::F16 => values
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect(),
        Dtype::BF16 => values
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect(),
        d => panic!("unsupported dtype for fixture: {:?}", d),
    };
    let view = TensorView::new(dtype, vec![values.len()], &bytes).unwrap();
    let mut tensors: HashMap<&str, TensorView<'_>> = HashMap::new();
    tensors.insert(name, view);
    let payload = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), payload).unwrap();
}

#[test]
fn mismatched_dtype_errors_by_default() {
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    write_one_tensor_model(&model_a, "tensor.weight", &values, Dtype::F16);
    write_one_tensor_model(&model_b, "tensor.weight", &values, Dtype::BF16);

    let config = MergeConfig {
        merge_method: MergeMethodConfig::Linear,
        models: vec![
            ModelConfig {
                model: model_a.to_string_lossy().into_owned(),
                parameters: Default::default(),
            },
            ModelConfig {
                model: model_b.to_string_lossy().into_owned(),
                parameters: Default::default(),
            },
        ],
        base_model: None,
        output_path: Some(workdir.path().join("out_strict")),
        dtype: "float16".to_string(),
        parameters: Default::default(),
        tokenizer: None,
        slices: None,
        allow_mixed_dtype: false,
        sanity: pmetal_merge::SanityLevel::default(),
        dry_run: false,
        ..Default::default()
    };

    let err = run_merge(&config).expect_err("should reject mixed-dtype inputs");
    match err {
        MergeError::DtypeMismatch { name, dtypes } => {
            assert_eq!(name, "tensor.weight");
            assert!(dtypes.iter().any(|s| s.contains("F16")));
            assert!(dtypes.iter().any(|s| s.contains("BF16")));
        }
        other => panic!("expected DtypeMismatch, got {:?}", other),
    }
}

#[test]
fn mismatched_dtype_passes_with_allow_mixed_flag() {
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    write_one_tensor_model(&model_a, "tensor.weight", &values, Dtype::F16);
    write_one_tensor_model(&model_b, "tensor.weight", &values, Dtype::BF16);

    let out = workdir.path().join("out_loose");
    let config = MergeConfig {
        merge_method: MergeMethodConfig::Linear,
        models: vec![
            ModelConfig {
                model: model_a.to_string_lossy().into_owned(),
                parameters: Default::default(),
            },
            ModelConfig {
                model: model_b.to_string_lossy().into_owned(),
                parameters: Default::default(),
            },
        ],
        base_model: None,
        output_path: Some(out.clone()),
        dtype: "float32".to_string(),
        parameters: Default::default(),
        tokenizer: None,
        slices: None,
        allow_mixed_dtype: true,
        sanity: pmetal_merge::SanityLevel::default(),
        dry_run: false,
        ..Default::default()
    };

    let path = run_merge(&config).expect("mixed dtypes should be allowed when opted in");
    assert_eq!(path, out);
    assert!(out.join("model.safetensors").exists());
}

#[test]
fn matching_dtype_passes_through() {
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    let values: Vec<f32> = (0..4).map(|i| i as f32).collect();
    write_one_tensor_model(&model_a, "tensor.weight", &values, Dtype::F16);
    write_one_tensor_model(&model_b, "tensor.weight", &values, Dtype::F16);

    let _ = MergeBuilder::new()
        .method(MergeMethodConfig::Linear)
        .add_model(model_a.to_string_lossy())
        .add_model(model_b.to_string_lossy())
        .output(workdir.path().join("out_match"))
        .dtype("float16")
        .run()
        .expect("matching f16 inputs should merge");
}
