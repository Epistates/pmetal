//! A DFlash draft conditions on hidden states tapped from the target. This
//! compares what the dynamic model and the native engine tap on a real Qwen3
//! checkpoint: cosine similarity per tapped layer, and the logits. Opt-in:
//!
//! ```text
//! PMETAL_QWEN3_DIR=<snapshot> PMETAL_DFLASH_LAYERS=1,9,17,25,33 \
//!     cargo test -p pmetal-models --test dflash_targets_real_weights -- --nocapture
//! ```

use std::path::PathBuf;

use pmetal_cache::speculative::SpecCapture;
use pmetal_data::Tokenizer;
use pmetal_models::DynamicModel;
use pmetal_models::dflash_decoder::DFlashTarget;
use pmetal_models::dflash_native_target::NativeQwen3Target;
use pmetal_runtime::compat::{Array, Dtype};
use serial_test::serial;

fn floats(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32.as_i32()).reshape(&[-1]);
    a.eval();
    a.as_slice::<f32>().to_vec()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[test]
#[serial]
fn the_dynamic_target_taps_what_the_native_one_does() {
    let Some(dir) = std::env::var_os("PMETAL_QWEN3_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_QWEN3_DIR is not set; skipping");
        return;
    };
    let layers: Vec<usize> = std::env::var("PMETAL_DFLASH_LAYERS")
        .unwrap_or_else(|_| "1,9,17,25,33".into())
        .split(',')
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let tokenizer = Tokenizer::from_model_dir(&dir).unwrap();
    let ids: Vec<i32> = tokenizer
        .encode("Explain in detail how a binary search tree works, with a code example in Python.")
        .unwrap()
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let input = Array::from_slice(&ids, &[1, ids.len() as i32]);

    let mut dynamic = DynamicModel::load(&dir).unwrap();
    let mut dyn_capture = SpecCapture::with_layers(layers.clone());
    let mut kv = DFlashTarget::make_kv_cache(&dynamic, 256);
    let mut mamba = DFlashTarget::make_mamba_cache(&dynamic);
    let dyn_logits = dynamic
        .forward_with_capture(
            &input,
            None,
            Some(&mut kv),
            mamba.as_mut(),
            &mut dyn_capture,
        )
        .unwrap();

    let mut native = NativeQwen3Target::load(&dir).unwrap();
    let mut nat_capture = SpecCapture::with_layers(layers.clone());
    let nat_logits = native
        .forward_with_capture(&input, None, None, None, &mut nat_capture)
        .unwrap();
    pmetal_runtime::check_unobserved_error().unwrap();

    for layer in &layers {
        let (d, n) = (
            floats(&dyn_capture.hidden_states[layer]),
            floats(&nat_capture.hidden_states[layer]),
        );
        eprintln!("layer {layer}: cosine {:.5}", cosine(&d, &n));
    }
    let last = |logits: &Array| {
        let v = floats(logits);
        let vocab = v.len() / ids.len();
        v[v.len() - vocab..].to_vec()
    };
    eprintln!(
        "last-position logits: cosine {:.5}",
        cosine(&last(&dyn_logits), &last(&nat_logits))
    );
}
