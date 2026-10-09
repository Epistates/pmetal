//! An adapted model decodes what its own forward pass computes.
//!
//! Decoding one token at a time against a cache runs paths the uncached
//! forward never takes: fused decode projections, concatenated weights, a
//! whole separate engine for Qwen 3.5. A fast path that multiplies by a
//! projection's stored weight skips the adapter on it, so the adapted model
//! generates as the base model would while training and scoring see the
//! adapter: GRPO's rollouts then come from a different policy than the one
//! its log-probabilities are taken from, and `infer --lora` without fusing
//! ignores the adapter.
//!
//! Each architecture gets an adapter on every projection, with nonzero `B`,
//! and decodes a few greedy steps with its caches; after every step the
//! uncached forward over the same prefix has to agree.

mod common;

use std::collections::HashMap;
use std::rc::Rc;

use common::cases;
use pmetal_bridge::compat::{Array, Dtype, random};
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, QLoraConfig, quantize_base};
use pmetal_models::dispatcher::DynamicModel;

const PROMPT: [i32; 6] = [3, 11, 7, 2, 9, 5];
const STEPS: usize = 3;

fn ids(tokens: &[i32]) -> Array {
    Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32])
}

fn last_row(logits: &Array) -> Vec<f32> {
    let seq = logits.dim(1);
    let vocab = logits.dim(2);
    let mut row = logits
        .slice(&[0, seq - 1, 0], &[1, seq, vocab])
        .as_dtype(Dtype::Float32.as_i32())
        .reshape(&[vocab]);
    row.to_f32_vec(vocab as usize).expect("row")
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// How far the cached decode of `config_json`, adapted, strays from the
/// uncached forward, relative to the logits' scale; `Err` names the first
/// step where they pick different tokens. `scale_b` multiplies the adapters'
/// random `B`: 1 adapts the model, 0 leaves it computing the base model.
/// `packed` packs the base for QLoRA first.
fn worst_disagreement(
    name: &str,
    config_json: &str,
    scale_b: f32,
    packed: bool,
) -> Result<f32, String> {
    random::seed(3);
    let mut base = DynamicModel::from_config(config_json).expect("build");
    if packed {
        quantize_base(&mut base, &QLoraConfig::default()).expect("pack");
    }
    let lora = LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        ..Default::default()
    };
    let mut model = AdaptedModel::attach(base, lora).expect("attach");
    let nudged: HashMap<Rc<str>, Array> = model
        .lora_parameters()
        .into_iter()
        .map(|(key, p)| {
            let value = random::uniform_range(-0.1, 0.1, p.shape(), Dtype::Float32).multiply(
                &Array::from_f32(if key.ends_with("lora_b") {
                    scale_b
                } else {
                    1.0
                }),
            );
            (key, value)
        })
        .collect();
    model.set_lora_parameters(&nudged);

    let capacity = PROMPT.len() + STEPS + 1;
    let mut cache = model.create_cache(capacity);
    let mut mamba = model.model().create_mamba_cache();
    let mut step = |model: &mut AdaptedModel, tokens: &[i32]| -> Array {
        let dynamic = model.model_mut();
        match mamba.as_mut() {
            Some(mamba) => dynamic
                .forward_with_hybrid_cache(&ids(tokens), None, Some(&mut cache), Some(mamba))
                .expect("cached step"),
            None => dynamic
                .forward_with_cache(&ids(tokens), None, Some(&mut cache))
                .expect("cached step"),
        }
    };

    let mut prefix = PROMPT.to_vec();
    let mut cached = last_row(&step(&mut model, &prefix));
    let mut worst = 0.0f32;
    for n in 0..STEPS {
        let fresh = last_row(&model.forward(&ids(&prefix), None).expect("forward"));
        pmetal_bridge::check_unobserved_error().map_err(|e| format!("{name}: {e}"))?;
        let scale = fresh.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let diff = cached
            .iter()
            .zip(&fresh)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(diff / scale);
        if argmax(&cached) != argmax(&fresh) {
            return Err(format!(
                "{name} step {n}: the cached decode picks token {} and the forward {} \
                 (relative difference {:e})",
                argmax(&cached),
                argmax(&fresh),
                diff / scale
            ));
        }
        let next = argmax(&cached) as i32;
        prefix.push(next);
        cached = last_row(&step(&mut model, &[next]));
    }
    Ok(worst)
}

/// Architectures whose cached decode disagrees with their forward even with
/// no adapter attached (`B` zero): a cache defect of their own, which this
/// test can't tell apart from an adapter being skipped. Each is checked to
/// still fail that way, so the entry goes when it is fixed.
const BASE_DECODE_DEFECTS: &[&str] = &[];

#[test]
fn an_adapted_model_decodes_what_its_forward_computes() {
    let mut problems = Vec::new();
    for case in cases() {
        if BASE_DECODE_DEFECTS.contains(&case.name) {
            assert!(
                worst_disagreement(case.name, case.config_json, 0.0, false)
                    .map_or(true, |worst| worst >= 1e-3),
                "{}: its unadapted decode agrees with its forward now; drop it from \
                 BASE_DECODE_DEFECTS",
                case.name
            );
            continue;
        }
        for packed in [false, true] {
            match worst_disagreement(case.name, case.config_json, 1.0, packed) {
                Ok(worst) if worst < 1e-3 => {}
                Ok(worst) => problems.push(format!(
                    "{}{}: cached decode off from the forward by {worst:e} (relative)",
                    case.name,
                    if packed { " (QLoRA)" } else { "" }
                )),
                Err(e) => problems.push(if packed { format!("{e} (QLoRA)") } else { e }),
            }
        }
    }
    assert!(
        problems.is_empty(),
        "adapted models whose decode ignores their adapters:\n  {}",
        problems.join("\n  ")
    );
}
