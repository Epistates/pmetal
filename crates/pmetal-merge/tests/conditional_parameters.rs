//! Filter-conditional parameters (`weight: [{value, filter}, ...]`) must be
//! resolved against the name of the tensor being merged, in every method.

mod common;

use common::{assert_close, read_tensor, write_model};
use pmetal_merge::{MergeConfig, MergeMethodConfig, ModelConfig, run_merge};
use safetensors::Dtype;

const ATTN: &str = "model.layers.0.self_attn.q_proj.weight";
const MLP: &str = "model.layers.0.mlp.up_proj.weight";

fn two_models(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let a = dir.join("a");
    let b = dir.join("b");
    write_model(
        &a,
        &[
            (ATTN, vec![2.0, 4.0], vec![2]),
            (MLP, vec![2.0, 4.0], vec![2]),
        ],
        Dtype::F32,
    );
    write_model(
        &b,
        &[
            (ATTN, vec![4.0, 8.0], vec![2]),
            (MLP, vec![4.0, 8.0], vec![2]),
        ],
        Dtype::F32,
    );
    (a, b)
}

fn config(
    method: MergeMethodConfig,
    a: &std::path::Path,
    b: &std::path::Path,
    out: &std::path::Path,
    yaml_params_a: &str,
) -> MergeConfig {
    let model_a: ModelConfig = serde_saphyr::from_str(&format!(
        "model: {}\nparameters:\n{yaml_params_a}",
        a.display()
    ))
    .unwrap();
    MergeConfig {
        merge_method: method,
        models: vec![
            model_a,
            ModelConfig {
                model: b.to_string_lossy().into_owned(),
                parameters: Default::default(),
            },
        ],
        output_path: Some(out.to_path_buf()),
        dtype: "float32".into(),
        ..Default::default()
    }
}

#[test]
fn linear_weight_filter_applies_per_tensor() {
    let work = tempfile::tempdir().unwrap();
    let (a, b) = two_models(work.path());
    let out = work.path().join("out");
    // Model A counts fully in attention tensors and not at all in MLP ones.
    let cfg = config(
        MergeMethodConfig::Linear,
        &a,
        &b,
        &out,
        "  weight:\n    - value: 1.0\n      filter: self_attn\n    - value: 0.0\n      filter: mlp\n",
    );
    run_merge(&cfg).unwrap();

    // attention: (1*A + 1*B) / 2; mlp: (0*A + 1*B) / 1
    assert_close(&read_tensor(&out, ATTN), &[3.0, 6.0], 1e-6);
    assert_close(&read_tensor(&out, MLP), &[4.0, 8.0], 1e-6);
}

#[test]
fn slerp_t_filter_applies_per_tensor() {
    let work = tempfile::tempdir().unwrap();
    let (a, b) = two_models(work.path());
    let out = work.path().join("out");
    // t = 0 keeps A on attention tensors, t = 1 keeps B on MLP ones.
    let cfg = config(
        MergeMethodConfig::Slerp,
        &a,
        &b,
        &out,
        "  t:\n    - value: 0.0\n      filter: self_attn\n    - value: 1.0\n      filter: mlp\n",
    );
    run_merge(&cfg).unwrap();

    assert_close(&read_tensor(&out, ATTN), &[2.0, 4.0], 1e-6);
    assert_close(&read_tensor(&out, MLP), &[4.0, 8.0], 1e-6);
}
