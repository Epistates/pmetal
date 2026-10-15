//! Shared helpers for the integration tests: fixture loading and numeric checks.
//!
//! Fixtures are small safetensors files dumped from the reference BigVGAN
//! implementation (random weights, fixed seeds), so every test compares our
//! numbers against the reference's, not against shapes.

#![allow(dead_code)]

use pmetal_runtime::compat::Array;
use std::collections::HashMap;
use std::path::PathBuf;

/// One safetensors fixture: named f32 tensors plus the file's string metadata.
pub struct Fixture {
    tensors: HashMap<String, (Vec<i32>, Vec<f32>)>,
    metadata: HashMap<String, String>,
}

impl Fixture {
    pub fn path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(format!("{name}.safetensors"))
    }

    pub fn load(name: &str) -> Self {
        Self::load_path(&Self::path(name))
    }

    pub fn load_path(path: &std::path::Path) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("valid safetensors");
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).expect("metadata");
        let mut tensors = HashMap::new();
        for (key, view) in st.tensors() {
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{key}");
            let shape: Vec<i32> = view.shape().iter().map(|&d| d as i32).collect();
            let data: Vec<f32> = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            tensors.insert(key, (shape, data));
        }
        let metadata = meta.metadata().clone().unwrap_or_default();
        Self { tensors, metadata }
    }

    /// The tensor as an array with its original (PyTorch) shape.
    pub fn array(&self, key: &str) -> Array {
        let (shape, data) = self.get(key);
        Array::from_f32_slice(data, shape)
    }

    /// The tensors under `prefix.` as a state dict with the prefix stripped
    /// (an empty prefix takes everything).
    pub fn state_dict(&self, prefix: &str) -> pmetal_vocoder::nn::StateDict {
        let want = if prefix.is_empty() {
            String::new()
        } else {
            format!("{prefix}.")
        };
        let map = self
            .tensors
            .iter()
            .filter_map(|(k, (shape, data))| {
                k.strip_prefix(&want)
                    .map(|name| (name.to_owned(), Array::from_f32_slice(data, shape)))
            })
            .collect();
        pmetal_vocoder::nn::StateDict::new(map)
    }

    /// Every tensor except the named ones, as a state dict.
    pub fn state_dict_except(&self, skip: &[&str]) -> pmetal_vocoder::nn::StateDict {
        let map = self
            .tensors
            .iter()
            .filter(|(k, _)| !skip.contains(&k.as_str()))
            .map(|(k, (shape, data))| (k.clone(), Array::from_f32_slice(data, shape)))
            .collect();
        pmetal_vocoder::nn::StateDict::new(map)
    }

    pub fn shape(&self, key: &str) -> &[i32] {
        &self.get(key).0
    }

    pub fn values(&self, key: &str) -> &[f32] {
        &self.get(key).1
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// A string metadata entry written by the dump script (JSON text).
    pub fn meta(&self, key: &str) -> &str {
        self.metadata
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no metadata `{key}`"))
    }

    fn get(&self, key: &str) -> &(Vec<i32>, Vec<f32>) {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no tensor `{key}`"))
    }
}

/// Fail on any error a runtime op recorded without throwing a Rust error.
pub fn assert_runtime_ok() {
    pmetal_runtime::check_unobserved_error().expect("runtime op recorded an error");
    pmetal_runtime::check_last_error().expect("runtime op recorded an error");
}

/// Row-major f32 contents of `array` (evaluates it, and checks the runtime).
pub fn to_vec(array: &Array) -> Vec<f32> {
    let array = array.as_dtype(pmetal_runtime::compat::Dtype::Float32);
    let v = array.as_slice::<f32>().to_vec();
    assert_runtime_ok();
    v
}

/// Largest `|a - b|` over the common length (panics on a length mismatch).
pub fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

/// Assert `|actual - expected| <= atol + rtol * |expected|` elementwise and
/// that the shapes agree, reporting the worst offender.
pub fn assert_close(
    actual: &Array,
    expected_shape: &[i32],
    expected: &[f32],
    rtol: f32,
    atol: f32,
    what: &str,
) {
    assert_eq!(actual.shape(), expected_shape, "{what}: shape");
    let got = to_vec(actual);
    assert_close_slice(&got, expected, rtol, atol, what);
}

pub fn assert_close_slice(got: &[f32], expected: &[f32], rtol: f32, atol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    let mut worst = (0usize, 0.0f32);
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite value {g} at {i}");
        let excess = (g - e).abs() - (atol + rtol * e.abs());
        if excess > worst.1 {
            worst = (i, excess);
        }
    }
    assert!(
        worst.1 <= 0.0,
        "{what}: element {} differs: got {} expected {} (|diff| {}, rtol {rtol}, atol {atol})",
        worst.0,
        got[worst.0],
        expected[worst.0],
        (got[worst.0] - expected[worst.0]).abs(),
    );
}

/// Total number of scalar parameters of a module (counts `weight_v` and
/// `weight_g` separately, like `sum(p.numel() for p in module.parameters())`).
pub fn count_params<M: pmetal_runtime::compat::ModuleParameters>(module: &M) -> usize {
    use pmetal_runtime::compat::NestedValue;
    fn walk(node: &NestedValue<&Array>) -> usize {
        match node {
            NestedValue::Value(a) => a.size(),
            NestedValue::Map(m) => m.values().map(walk).sum(),
        }
    }
    module.parameters().values().map(walk).sum()
}
