//! A bf16 base stays bf16 under adapters.
//!
//! The adapters are f32 (they are trained, and rounding them to bf16 would
//! lose the small steps), the base is bf16, and the model must still compute
//! in bf16: logits come back bf16, a merge leaves the folded weights bf16 and
//! the merged forward agrees with the adapted one to bf16 resolution, and
//! training keeps the adapters f32 and finite.

mod common;

use common::{cases, config_int, input_ids};
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Dtype, ModuleParametersExt, random};

/// Dense, MoE, state-space, hybrid and gated-delta-net models.
const KINDS: [&str; 7] = [
    "llama",
    "qwen3_moe",
    "deepseek",
    "nemotron_h",
    "granitemoehybrid",
    "qwen3_next",
    "gemma4",
];

fn bf16_model(name: &str) -> (AdaptedModel, i32) {
    let case = cases().into_iter().find(|c| c.name == name).unwrap();
    random::seed(71);
    let mut base = DynamicModel::from_config(case.config_json).expect("build");
    for (_, param) in base.flatten_params_mut() {
        *param = param.as_dtype(Dtype::Bfloat16.as_i32());
    }
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        ..Default::default()
    };
    (
        AdaptedModel::attach(base, config).expect("attach"),
        config_int(&case, "vocab_size").unwrap(),
    )
}

fn drain(name: &str) {
    if let Err(e) = pmetal_runtime::check_unobserved_error() {
        panic!("{name}: a runtime op threw: {e}");
    }
}

#[test]
fn a_bf16_base_computes_in_bf16_and_keeps_f32_adapters() {
    for name in KINDS {
        let (mut model, vocab) = bf16_model(name);
        let ids = input_ids(vocab);
        let logits = model.forward(&ids, None).unwrap();
        assert_eq!(logits.dtype(), Dtype::Bfloat16, "{name}: logits");
        let params = model.lora_parameters();
        assert!(!params.is_empty(), "{name}: no adapters");
        for (key, value) in &params {
            assert_eq!(value.dtype(), Dtype::Float32, "{name}: {key}");
        }
        drain(name);
    }
}

#[test]
fn merging_into_a_bf16_base_keeps_it_bf16_and_the_output() {
    for name in KINDS {
        let (mut model, vocab) = bf16_model(name);
        let ids = input_ids(vocab);
        // Nonzero adapters, so the merge has something to fold.
        random::seed(3);
        let set = model
            .lora_parameters()
            .into_iter()
            .map(|(k, p)| {
                let fresh = random::uniform_range(-0.05, 0.05, p.shape(), Dtype::Float32);
                (k, fresh)
            })
            .collect();
        model.set_lora_parameters(&set);
        let adapted = model.forward(&ids, None).unwrap();

        model.merge().unwrap_or_else(|e| panic!("{name}: {e}"));
        let merged = model.forward(&ids, None).unwrap();
        assert_eq!(merged.dtype(), Dtype::Bfloat16, "{name}");
        // Folding rounds each weight to bf16 once; the logits agree to the
        // resolution of bf16 at their size.
        let scale = adapted
            .as_dtype(Dtype::Float32.as_i32())
            .abs()
            .max(None)
            .item_f32();
        let diff = max_abs_diff(&adapted, &merged);
        assert!(diff <= 0.05 * scale.max(1.0), "{name}: merge moved the logits by {diff} (scale {scale})");

        model.fuse().unwrap_or_else(|e| panic!("{name}: {e}"));
        for (key, weight) in model.model().flatten_params() {
            if key.ends_with(".weight") && weight.ndim() >= 2 {
                assert_ne!(weight.dtype(), Dtype::Float32, "{name}: fused {key} is f32");
            }
        }
        drain(name);
    }
}

#[test]
fn training_a_bf16_base_keeps_the_adapters_f32_and_finite() {
    for name in KINDS {
        let (mut model, vocab) = bf16_model(name);
        let ids = input_ids(vocab);
        let mut optimizer = pmetal_runtime::compat::optimizers::AdamW::new(1e-3, 0.0);
        let losses: Vec<f32> = (0..3)
            .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
            .collect();
        assert!(losses.iter().all(|l| l.is_finite()), "{name}: {losses:?}");
        for (key, value) in model.lora_parameters() {
            assert_eq!(value.dtype(), Dtype::Float32, "{name}: {key} after training");
        }
        drain(name);
    }
}
