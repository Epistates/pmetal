//! A full fine-tune trains every parameter, and each one gets the gradient
//! its loss says it should.
//!
//! The recurring way a trainer goes wrong is silent: a parameter sits in the
//! forward pass and trains on nothing, because a host readback, a Metal buffer
//! or a raw weight read cut autograd's graph on the way to it. The forward and
//! the loss are unchanged, so nothing complains. This takes one step in
//! training mode the way the trainer does, over every parameter of every
//! architecture (the embedding, every norm, the output head, tied or not), and
//! requires a finite, nonzero gradient on each; and holds the gradient of the
//! parameters a cut is most likely to hide against a finite difference of the
//! loss.

mod common;

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use common::{SEQ_LEN, cases, input_ids, train_step};
use pmetal_lora::{AdaptedModel, FullFineTune, TrainableModel};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::optimizers::AdamW;
use pmetal_runtime::compat::{Array, Dtype, ModuleParameters, ModuleParametersExt, random};
use pmetal_runtime::inline_array::value_and_grad;
use pmetal_runtime::nn::with_training_mode;

const SCORED_TAIL: i32 = 4;

fn vocab_of(config_json: &str) -> i32 {
    let value: serde_json::Value = serde_json::from_str(config_json).expect("config json");
    value["vocab_size"].as_i64().expect("vocab_size") as i32
}

fn drain(case: &str) {
    if let Err(e) = pmetal_runtime::check_unobserved_error() {
        panic!("{case}: a bridge op threw: {e}");
    }
}

/// Fixed weights over the last few positions' logits, so the loss is O(1).
fn loss_weights(vocab: i32) -> Array {
    let mut w = vec![0.0_f32; (SEQ_LEN * vocab) as usize];
    for (i, x) in w
        .iter_mut()
        .enumerate()
        .skip(((SEQ_LEN - SCORED_TAIL) * vocab) as usize)
    {
        *x = ((i * 37 % 101) as f32 / 101.0) - 0.5;
    }
    Array::from_f32_slice(&w, &[1, SEQ_LEN, vocab])
}

fn loss_of(logits: &Array, weights: &Array) -> Array {
    logits
        .as_dtype(Dtype::Float32.as_i32())
        .multiply(weights)
        .sum_all()
}

/// The loss and the gradient of every trainable parameter, in training mode.
fn gradients(
    model: &mut AdaptedModel,
    ids: &Array,
    weights: &Array,
) -> (f32, BTreeMap<String, Array>) {
    let live = model.lora_parameters();
    let mut names: Vec<Rc<str>> = live.keys().cloned().collect();
    names.sort();
    let params: Vec<Array> = names.iter().map(|n| live[n].clone()).collect();
    let (loss, grads) = with_training_mode(|| {
        let (loss, grads) = value_and_grad(
            |arrays| {
                let restored: HashMap<Rc<str>, Array> =
                    names.iter().cloned().zip(arrays.iter().cloned()).collect();
                model.set_lora_parameters(&restored);
                loss_of(&model.forward(ids, None).unwrap(), weights)
            },
            &params,
            &[],
        );
        loss.eval();
        for g in &grads {
            g.eval();
        }
        Ok((loss.item_f32(), grads))
    })
    .unwrap();
    model.set_lora_parameters(&live);
    (
        loss,
        names
            .iter()
            .map(|n| n.to_string())
            .zip(grads)
            .collect::<BTreeMap<_, _>>(),
    )
}

fn full_model(config_json: &str, options: FullFineTune) -> AdaptedModel {
    random::seed(3);
    let base = DynamicModel::from_config(config_json).expect("builds");
    AdaptedModel::full(base, options).expect("a dense model trains in full")
}

fn norm_of(grad: &Array) -> f32 {
    grad.as_dtype(Dtype::Float32.as_i32())
        .square()
        .sum_all()
        .sqrt()
        .item_f32()
}

#[test]
fn every_parameter_of_every_architecture_gets_a_nonzero_gradient() {
    let mut problems = Vec::new();
    let mut checked = 0;
    for case in cases() {
        let mut model = full_model(case.config_json, FullFineTune::default());
        let vocab = vocab_of(case.config_json);
        let (loss, grads) = gradients(&mut model, &input_ids(vocab), &loss_weights(vocab));
        drain(case.name);
        if !loss.is_finite() {
            problems.push(format!("{}: loss is {loss}", case.name));
            continue;
        }
        // Every parameter the model holds is trainable, and so has a gradient.
        let all: usize = model.model().flatten_params().len();
        if grads.len() != all {
            problems.push(format!(
                "{}: {} of the model's {all} parameters are trainable",
                case.name,
                grads.len()
            ));
        }
        for (name, grad) in &grads {
            let norm = norm_of(grad);
            if !(norm.is_finite() && norm > 0.0) {
                problems.push(format!("{}: {name} has gradient norm {norm}", case.name));
            }
        }
        checked += 1;
    }
    assert!(checked >= 19, "only {checked} architectures were exercised");
    assert!(
        problems.is_empty(),
        "parameters that train on nothing:\n  {}",
        problems.join("\n  ")
    );
}

/// The embedding, the final norm and the output head (the first weight the
/// loss reads and the last two) carry the gradient the loss says: the
/// derivative of the loss along their own gradient is its norm.
#[test]
fn the_gradients_most_easily_cut_agree_with_a_finite_difference() {
    const H: f32 = 1e-2;
    let mut problems = Vec::new();
    // A tied head (qwen3) and an untied one (llama).
    for case in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "llama" | "qwen3"))
    {
        let mut model = full_model(case.config_json, FullFineTune::default());
        let vocab = vocab_of(case.config_json);
        let (ids, weights) = (input_ids(vocab), loss_weights(vocab));
        let (loss, grads) = gradients(&mut model, &ids, &weights);
        let loss_rounding = 128.0 * f32::EPSILON * loss.abs().max(1.0) / H;
        let base = model.lora_parameters();
        for key in grads.keys().filter(|k| {
            k.ends_with("embed_tokens.weight")
                || k.ends_with("lm_head.weight")
                || *k == "model.norm.weight"
        }) {
            let grad = grads[key].as_dtype(Dtype::Float32.as_i32());
            let norm = norm_of(&grad);
            let direction = grad.divide(&Array::from_f32(norm));
            let mut at = |step: f32| {
                let mut params = base.clone();
                let moved = base[key.as_str()].add(&direction.multiply(&Array::from_f32(step)));
                params.insert(Rc::from(key.as_str()), moved);
                model.set_lora_parameters(&params);
                let value = with_training_mode(|| {
                    Ok(loss_of(&model.forward(&ids, None).unwrap(), &weights).item_f32())
                })
                .unwrap();
                drain(case.name);
                value
            };
            let central = |h: f32, at: &mut dyn FnMut(f32) -> f32| (at(h) - at(-h)) / (2.0 * h);
            let steps = [
                central(H, &mut at),
                central(H / 2.0, &mut at),
                central(H / 4.0, &mut at),
            ];
            model.set_lora_parameters(&base);
            let fd = [
                (4.0 * steps[1] - steps[0]) / 3.0,
                (4.0 * steps[2] - steps[1]) / 3.0,
            ]
            .into_iter()
            .min_by(|a, b| (a - norm).abs().total_cmp(&(b - norm).abs()))
            .unwrap();
            if (fd - norm).abs() > 2e-2 * norm + loss_rounding {
                problems.push(format!(
                    "{}: {key} autograd |g| {norm} vs finite difference along g {fd}",
                    case.name
                ));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "gradients that disagree with the loss:\n  {}",
        problems.join("\n  ")
    );
}

#[test]
fn the_trainers_view_of_what_trains_is_the_checkpoints() {
    for case in cases().into_iter().filter(|c| c.name == "qwen3") {
        let model = full_model(case.config_json, FullFineTune::default());
        let mut trainer_keys: Vec<String> = model
            .trainable_parameters()
            .keys()
            .map(|k| k.to_string())
            .collect();
        let mut checkpoint_keys: Vec<String> = TrainableModel::lora_parameters(&model)
            .keys()
            .map(|k| k.to_string())
            .collect();
        trainer_keys.sort();
        checkpoint_keys.sort();
        assert_eq!(trainer_keys, checkpoint_keys);
        let elements: usize = model
            .model()
            .flatten_params()
            .values()
            .map(|p| p.shape().iter().map(|&d| d as usize).product::<usize>())
            .sum();
        assert_eq!(TrainableModel::num_trainable_params(&model), elements);
        assert!(model.is_full_finetune() && model.adapted_projections().is_empty());
        assert!(TrainableModel::adapter_target_modules(&model).is_empty());
    }
}

#[test]
fn a_frozen_embedding_keeps_its_value_and_leaves_the_rest_training() {
    for case in cases()
        .into_iter()
        .filter(|c| matches!(c.name, "llama" | "qwen3"))
    {
        let tied = case.name == "qwen3";
        let mut model = full_model(
            case.config_json,
            FullFineTune {
                freeze_embeddings: true,
            },
        );
        let frozen = model.frozen_parameters();
        assert_eq!(frozen, ["model.embed_tokens.weight"], "{}", case.name);
        let keys: Vec<String> = model
            .lora_parameters()
            .keys()
            .map(|k| k.to_string())
            .collect();
        assert!(!keys.iter().any(|k| k == "model.embed_tokens.weight"));
        // An untied head still trains; a tied one is the embedding.
        assert_eq!(
            keys.iter().any(|k| k.ends_with("lm_head.weight")),
            !tied,
            "{}",
            case.name
        );
        assert!(
            !model
                .trainable_parameters()
                .contains_key("model.embed_tokens.weight")
        );

        let before = model
            .model()
            .flatten_params()
            .remove("model.embed_tokens.weight")
            .unwrap()
            .clone();
        let ids = input_ids(vocab_of(case.config_json));
        let mut optimizer = AdamW::new(1e-2, 0.0);
        for _ in 0..3 {
            train_step(&mut model, &mut optimizer, &ids);
        }
        let after = model
            .model()
            .flatten_params()
            .remove("model.embed_tokens.weight")
            .unwrap()
            .clone();
        let moved = before.subtract(&after).abs().mean_all().item_f32();
        assert_eq!(moved, 0.0, "{}: the frozen embedding moved", case.name);
        // ...while something else did.
        let layer = "model.layers.0.mlp.down_proj.weight";
        let first = model.model().flatten_params()[layer].clone();
        train_step(&mut model, &mut optimizer, &ids);
        let second = model.model().flatten_params()[layer].clone();
        assert!(first.subtract(&second).abs().mean_all().item_f32() > 0.0);
    }
}

#[test]
fn a_quantized_model_cannot_be_fully_fine_tuned() {
    use pmetal_lora::{QLoraConfig, QLoraScheme, quantize_base};
    let case = cases().into_iter().find(|c| c.name == "llama").unwrap();
    let mut base = DynamicModel::from_config(case.config_json).unwrap();
    let packed = quantize_base(
        &mut base,
        &QLoraConfig {
            scheme: QLoraScheme::Int8,
            group_size: 64,
            double_quant: false,
        },
    )
    .unwrap();
    assert!(packed.packed > 0);
    let err = AdaptedModel::full(base, FullFineTune::default()).unwrap_err();
    assert!(err.to_string().contains("dense"), "{err}");
}
