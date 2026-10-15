//! FP8 (E4M3) checkpoints with block-wise inverse scales
//! (`<name>.weight_scale_inv`, one scale per 128x128 block) merge as the
//! dequantized weights, and the scale tensors do not leak into the output.

mod common;

use common::{assert_close, read_tensor};
use pmetal_merge::{
    MergeConfig, MergeMethodConfig, ModelConfig, SafetensorsLoader, TensorLoader, run_merge,
};
use pmetal_runtime::compat::{Array, Dtype as RtDtype};
use safetensors::Dtype;
use safetensors::tensor::TensorView;
use std::collections::HashMap;
use std::path::Path;

const W: &str = "model.layers.0.mlp.up_proj.weight";

/// E4M3 bytes of `values` (exactly representable ones round-trip).
fn e4m3(values: &[f32]) -> Vec<u8> {
    let encoded = Array::from_f32_slice(values, &[values.len() as i32]).to_fp8();
    let widened = encoded.as_dtype(RtDtype::Uint32.as_i32());
    widened.as_slice::<u32>().iter().map(|&v| v as u8).collect()
}

/// A [128, 256] FP8 weight whose left block is `left` and right block `right`
/// (before scaling), with block scales `[s0, s1]`, plus a bf16 tensor.
fn write_fp8_model(dir: &Path, left: f32, right: f32, scales: [f32; 2], norm: f32) {
    std::fs::create_dir_all(dir).unwrap();
    let mut values = Vec::with_capacity(128 * 256);
    for _ in 0..128 {
        values.extend(std::iter::repeat_n(left, 128));
        values.extend(std::iter::repeat_n(right, 128));
    }
    let weight = e4m3(&values);
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let norm_bytes: Vec<u8> = [norm, norm]
        .iter()
        .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
        .collect();
    let views = HashMap::from([
        (
            W.to_string(),
            TensorView::new(Dtype::F8_E4M3, vec![128, 256], &weight).unwrap(),
        ),
        (
            format!("{W}_scale_inv"),
            TensorView::new(Dtype::F32, vec![1, 2], &scale_bytes).unwrap(),
        ),
        (
            "model.norm.weight".to_string(),
            TensorView::new(Dtype::BF16, vec![2], &norm_bytes).unwrap(),
        ),
    ]);
    std::fs::write(
        dir.join("model.safetensors"),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
}

#[test]
fn the_loader_applies_block_scales_and_hides_the_scale_tensors() {
    let work = tempfile::tempdir().unwrap();
    let dir = work.path().join("m");
    write_fp8_model(&dir, 4.0, 4.0, [0.5, 2.0], 1.0);

    let loader = SafetensorsLoader::new(&dir).unwrap();
    assert_eq!(loader.tensor_names(), vec![W, "model.norm.weight"]);
    assert!(!loader.has_tensor(&format!("{W}_scale_inv")));

    let mut weight = loader.load_tensor(W).unwrap();
    assert_eq!(weight.shape(), &[128, 256]);
    let values = weight.to_f32_vec(128 * 256).unwrap();
    pmetal_runtime::check_unobserved_error().unwrap();
    // Left block 4 * 0.5, right block 4 * 2 (first and last row, edges of the blocks).
    for row in [0usize, 127] {
        for (col, want) in [(0, 2.0), (127, 2.0), (128, 8.0), (255, 8.0)] {
            assert_eq!(values[row * 256 + col], want, "row {row} col {col}");
        }
    }
}

#[test]
fn fp8_models_merge_as_their_dequantized_weights() {
    let work = tempfile::tempdir().unwrap();
    let (a, b, out) = (
        work.path().join("a"),
        work.path().join("b"),
        work.path().join("out"),
    );
    write_fp8_model(&a, 4.0, 4.0, [0.5, 2.0], 1.0); // dequantizes to 2 | 8
    write_fp8_model(&b, 2.0, 1.0, [1.0, 4.0], 3.0); // dequantizes to 2 | 4
    let config = MergeConfig {
        merge_method: MergeMethodConfig::Linear,
        models: [&a, &b]
            .iter()
            .map(|d| ModelConfig {
                model: d.to_string_lossy().into_owned(),
                parameters: Default::default(),
            })
            .collect(),
        output_path: Some(out.clone()),
        dtype: "float32".into(),
        ..Default::default()
    };
    run_merge(&config).unwrap();

    let merged = read_tensor(&out, W);
    assert_close(
        &[merged[0], merged[127], merged[128], merged[255]],
        &[2.0, 2.0, 6.0, 6.0],
        1e-6,
    );
    assert_close(&read_tensor(&out, "model.norm.weight"), &[2.0, 2.0], 1e-2);
    // The scales were consumed, not merged into the output.
    let names = SafetensorsLoader::new(&out).unwrap().tensor_names();
    assert!(
        names.iter().all(|n| !n.ends_with("_scale_inv")),
        "{names:?}"
    );
}
