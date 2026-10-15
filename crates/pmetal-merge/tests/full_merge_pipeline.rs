//! End-to-end merge pipeline tests covering the Phase 2 robustness work:
//!
//!  * tied-embedding detection drops the duplicate `lm_head.weight`
//!  * tokenizer + config sidecars are copied into the output dir
//!  * post-merge sanity sweep aborts on a tensor full of NaNs
//!  * dry-run mode skips writing weights but still validates structure

use pmetal_merge::{
    MergeBuilder, MergeConfig, MergeMethodConfig, ModelConfig, SanityLevel, TensorLoader, run_merge,
};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

fn write_model_dir(dir: &Path, tensors: &[(&str, Vec<f32>, Vec<usize>)], cfg_extra: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let bytes_keep: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|(n, vals, shape)| {
            let bytes: Vec<u8> = vals
                .iter()
                .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
                .collect();
            (n.to_string(), bytes, shape.clone())
        })
        .collect();
    let views: HashMap<&str, TensorView<'_>> = bytes_keep
        .iter()
        .map(|(n, b, s)| {
            (
                n.as_str(),
                TensorView::new(Dtype::F16, s.clone(), b).unwrap(),
            )
        })
        .collect();
    let payload = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), payload).unwrap();
    let cfg = format!(
        "{{\"model_type\": \"test\", \"hidden_size\": 4{}}}",
        cfg_extra
    );
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{\"version\": \"1.0\"}").unwrap();
    std::fs::write(
        dir.join("special_tokens_map.json"),
        "{\"bos_token\": \"<s>\"}",
    )
    .unwrap();
}

#[test]
fn tied_lm_head_merged_once() {
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    let embed = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let head = embed.clone();
    let tensors = [
        ("model.embed_tokens.weight", embed, vec![2, 4]),
        ("lm_head.weight", head, vec![2, 4]),
    ];
    // tie_word_embeddings: true → alias should be dropped on save.
    write_model_dir(&model_a, &tensors, ", \"tie_word_embeddings\": true");
    write_model_dir(&model_b, &tensors, ", \"tie_word_embeddings\": true");

    let out = workdir.path().join("out_tied");
    let _ = MergeBuilder::new()
        .method(MergeMethodConfig::Linear)
        .add_model(model_a.to_string_lossy())
        .add_model(model_b.to_string_lossy())
        .output(out.clone())
        .dtype("float16")
        .run()
        .unwrap();

    // Reload and confirm only one of the two names is present.
    let names: Vec<String> = pmetal_merge::SafetensorsLoader::new(&out)
        .unwrap()
        .tensor_names();
    assert!(
        names.iter().any(|n| n == "model.embed_tokens.weight"),
        "embed should survive: {:?}",
        names
    );
    assert!(
        !names.iter().any(|n| n == "lm_head.weight"),
        "lm_head alias should be dropped on tie: {:?}",
        names
    );

    // Sidecars must be copied alongside the weights.
    assert!(out.join("config.json").exists(), "config.json must copy");
    assert!(
        out.join("tokenizer.json").exists(),
        "tokenizer.json must copy"
    );
    assert!(
        out.join("special_tokens_map.json").exists(),
        "special_tokens_map.json must copy"
    );
}

#[test]
fn dry_run_does_not_write_weights() {
    let workdir = tempfile::tempdir().unwrap();
    let model_a = workdir.path().join("a");
    let model_b = workdir.path().join("b");

    let v = vec![1.0_f32, 2.0, 3.0, 4.0];
    let tensors = [("model.embed_tokens.weight", v.clone(), vec![1, 4])];
    write_model_dir(&model_a, &tensors, "");
    write_model_dir(&model_b, &tensors, "");

    let out = workdir.path().join("out_dry");

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
        dtype: "float16".to_string(),
        parameters: Default::default(),
        tokenizer: None,
        slices: None,
        allow_mixed_dtype: false,
        sanity: SanityLevel::Quick,
        dry_run: true,
        align_moe_experts: false,
    };

    run_merge(&config).expect("dry run should succeed");
    assert!(
        !out.join("model.safetensors").exists(),
        "dry run must not write weights"
    );
    // Sidecar copy is also skipped in dry-run (no tensors written).
    assert!(
        !out.join("config.json").exists(),
        "dry run must not copy sidecars either"
    );
}

#[test]
fn sanity_quick_rejects_nan_tensor() {
    use pmetal_merge::{MergedTensorReport, SanityLevel, check_tensor};
    use pmetal_runtime::compat::Array;

    // Direct unit-style coverage of the sanity guard — useful when the merge
    // produces a NaN through a numerically degenerate parameter combination.
    let arr = Array::from_f32_slice(&[1.0_f32, f32::NAN, 3.0], &[3]);
    let err = check_tensor("layers.0.weight", &arr, SanityLevel::Quick).unwrap_err();
    assert!(err.to_string().contains("NaN"));

    let arr_clean = Array::from_f32_slice(&[1.0_f32, 2.0, 3.0], &[3]);
    let r = check_tensor("layers.0.weight", &arr_clean, SanityLevel::Full)
        .unwrap()
        .expect("full level reports stats");
    let _: &MergedTensorReport = &r;
    assert!(!r.is_corrupt());
    assert!(r.mean.is_some());
}
