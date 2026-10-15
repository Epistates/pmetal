//! Fixtures shared by the integration tests: write small synthetic models
//! and read merged outputs back, failing on any runtime error that a
//! throwing op would otherwise hide behind a placeholder array.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use pmetal_merge::{SafetensorsLoader, TensorLoader};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

/// A named tensor to write: `(name, values, shape)`.
pub type Spec<'a> = (&'a str, Vec<f32>, Vec<usize>);

fn pack(values: &[f32], dtype: Dtype) -> Vec<u8> {
    match dtype {
        Dtype::F32 => values.iter().flat_map(|f| f.to_le_bytes()).collect(),
        Dtype::F16 => values
            .iter()
            .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
            .collect(),
        Dtype::BF16 => values
            .iter()
            .flat_map(|f| half::bf16::from_f32(*f).to_le_bytes())
            .collect(),
        d => panic!("unsupported fixture dtype {d:?}"),
    }
}

/// Write `tensors` to `dir/<file>` as one safetensors file in `dtype`.
pub fn write_file(dir: &Path, file: &str, tensors: &[Spec<'_>], dtype: Dtype) {
    std::fs::create_dir_all(dir).unwrap();
    let packed: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
        .iter()
        .map(|(n, v, s)| (n.to_string(), pack(v, dtype), s.clone()))
        .collect();
    let views: HashMap<&str, TensorView<'_>> = packed
        .iter()
        .map(|(n, b, s)| (n.as_str(), TensorView::new(dtype, s.clone(), b).unwrap()))
        .collect();
    let payload = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join(file), payload).unwrap();
}

/// Write a model directory holding one `model.safetensors` and a minimal
/// `config.json`.
pub fn write_model(dir: &Path, tensors: &[Spec<'_>], dtype: Dtype) {
    write_file(dir, "model.safetensors", tensors, dtype);
    std::fs::write(
        dir.join("config.json"),
        "{\"model_type\": \"test\", \"hidden_size\": 4}",
    )
    .unwrap();
}

/// Read tensor `name` from the model at `dir` as f32, failing on a runtime
/// error.
pub fn read_tensor(dir: &Path, name: &str) -> Vec<f32> {
    let loader = SafetensorsLoader::new(dir).unwrap();
    let mut array = loader.load_tensor(name).unwrap();
    let n: usize = loader.tensor_shape(name).unwrap().iter().product();
    let values = array.to_f32_vec(n).expect("tensor readable as f32");
    pmetal_runtime::check_unobserved_error().expect("runtime op failed");
    values
}

/// Assert two slices agree element-wise within `tol`.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "index {i}: got {g}, want {w} (tol {tol})\n got:  {got:?}\n want: {want:?}"
        );
    }
}
