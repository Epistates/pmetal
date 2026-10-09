//! Every adapter on every architecture gets the gradient its loss says it
//! should.
//!
//! An adapter can be in the forward pass and still train on nothing. Reading
//! a tensor back to the host and rebuilding it, or building an output from a
//! Metal buffer, cuts autograd's graph there: the forward is unchanged, the
//! loss is unchanged, and everything upstream of the cut trains on a zero or
//! partial gradient. A projection whose weight a fused path reads directly
//! skips its adapter outright. `every_adapter_reaches_the_output` in
//! `base_parity.rs` catches only the last of those.
//!
//! This one adapts every projection the model exposes, gives every adapter a
//! nonzero value, takes one gradient step in training mode the way the
//! trainer does, and holds the autograd gradient of each adapter against a
//! central finite difference of the loss itself. A cut anywhere between an
//! adapter and the loss shows up as a mismatch on that adapter.

mod common;

use std::collections::HashMap;
use std::rc::Rc;

use common::{SEQ_LEN, cases, input_ids};
use pmetal_bridge::compat::{Array, Dtype, random};
use pmetal_bridge::inline_array::value_and_grad;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_mlx::kernels::with_training_mode;
use pmetal_models::dispatcher::DynamicModel;

/// Positions whose logits enter the loss. A few, so the loss stays O(1) and a
/// finite difference resolves its derivative in f32.
const SCORED_TAIL: i32 = 4;

/// Length of the central difference's step, taken along a whole adapter
/// tensor (rank 4 by a few dozen features, entries O(0.05), so a norm near
/// 0.5): a few percent of it, large against f32 rounding of the loss and
/// small enough that the loss stays close to quadratic over it.
const H: f32 = 1e-2;

/// Architectures beyond the shared cases: the Qwen 3.5 MoE layer (router,
/// routed experts and the gated shared expert), which the dense Qwen 3.5 case
/// never builds.
fn extra_cases() -> Vec<(&'static str, &'static str)> {
    vec![(
        "qwen3_next_moe",
        r#"{
            "model_type": "qwen3_next",
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "norm_topk_prob": true,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 64,
            "partial_rotary_factor": 0.25,
            "tie_word_embeddings": false
        }"#,
    )]
}

fn all_cases() -> Vec<(&'static str, &'static str)> {
    let mut all: Vec<_> = cases()
        .into_iter()
        .map(|case| (case.name, case.config_json))
        .collect();
    all.extend(extra_cases());
    all
}

fn vocab_of(config_json: &str) -> i32 {
    let value: serde_json::Value = serde_json::from_str(config_json).expect("config json");
    value["vocab_size"].as_i64().expect("vocab_size") as i32
}

fn drain(case: &str) {
    if let Err(e) = pmetal_bridge::check_last_error() {
        panic!("{case}: a bridge op threw: {e}");
    }
}

/// Every projection, as the QLoRA paper adapts them.
fn lora_config() -> LoraConfig {
    LoraConfig {
        r: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        ..Default::default()
    }
}

/// Give both factors of every adapter a nonzero value, so each has a
/// gradient (`B` starts at zero, which zeroes `A`'s).
fn randomize_adapters(model: &mut AdaptedModel) {
    random::seed(7);
    let params: HashMap<Rc<str>, Array> = model
        .lora_parameters()
        .into_iter()
        .map(|(name, p)| {
            let r = random::uniform_range(-0.05, 0.05, p.shape(), Dtype::Float32);
            (name, r)
        })
        .collect();
    model.set_lora_parameters(&params);
}

/// Fixed weights over the last few positions' logits.
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

/// The loss at the current adapter values, in training mode.
fn loss_at(model: &mut AdaptedModel, ids: &Array, weights: &Array) -> f32 {
    with_training_mode(|| Ok(loss_of(&model.forward(ids, None).unwrap(), weights).item_f32()))
        .unwrap()
}

/// One step the way the trainer takes it: training mode on, no mask.
fn gradients(
    model: &mut AdaptedModel,
    names: &[Rc<str>],
    ids: &Array,
    weights: &Array,
) -> (f32, HashMap<Rc<str>, Array>) {
    let live = model.lora_parameters();
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
    (loss, names.iter().cloned().zip(grads).collect())
}

/// Problems with one architecture's adapter gradients, empty when it is right.
fn check_architecture(name: &str, config_json: &str) -> Vec<String> {
    random::seed(3);
    let Ok(base) = DynamicModel::from_config(config_json) else {
        return vec![format!("{name}: does not build")];
    };
    let mut model = AdaptedModel::attach(base, lora_config()).expect("attach");
    randomize_adapters(&mut model);
    let vocab = vocab_of(config_json);
    let ids = input_ids(vocab);
    let weights = loss_weights(vocab);

    let mut names: Vec<Rc<str>> = model.lora_parameters().into_keys().collect();
    names.sort();
    let (loss, grads) = gradients(&mut model, &names, &ids, &weights);
    drain(name);
    if !loss.is_finite() {
        return vec![format!("{name}: loss is {loss}")];
    }

    // Along the gradient's own direction `u = g / |g|`, the loss changes at
    // rate `<true gradient, u>`, which is `|g|` exactly when autograd has it
    // right. A missing term shows up as a shortfall (or excess), and the
    // whole tensor's signal goes into one difference, well clear of f32
    // rounding.
    let base = model.lora_parameters();
    let rounding = 16.0 * f32::EPSILON * loss.abs().max(1.0) / H;
    let mut problems = Vec::new();
    for key in &names {
        let g = grads[key].as_dtype(Dtype::Float32.as_i32());
        let norm = g.square().sum_all().sqrt().item_f32();
        if !(norm.is_finite() && norm > 0.0) {
            problems.push(format!("{name}: {key} has gradient norm {norm}"));
            continue;
        }
        let direction = g.divide(&Array::from_f32(norm));
        let mut nudged = |step: f32| {
            let mut params = base.clone();
            let moved = base[key].add(&direction.multiply(&Array::from_f32(step)));
            params.insert(key.clone(), moved);
            model.set_lora_parameters(&params);
            let loss = loss_at(&mut model, &ids, &weights);
            drain(name);
            loss
        };
        let fd = (nudged(H) - nudged(-H)) / (2.0 * H);
        model.set_lora_parameters(&base);
        if (fd - norm).abs() > 2e-2 * norm + rounding {
            problems.push(format!(
                "{name}: {key} autograd |g| {norm} vs finite difference along g {fd}"
            ));
        }
    }
    problems
}

#[test]
fn every_adapter_gets_its_gradient() {
    let mut problems = Vec::new();
    let mut checked = 0;
    for (name, config_json) in all_cases() {
        problems.extend(check_architecture(name, config_json));
        checked += 1;
    }
    assert!(checked >= 19, "only {checked} architectures were exercised");
    assert!(
        problems.is_empty(),
        "adapter gradients that disagree with the loss:\n  {}",
        problems.join("\n  ")
    );
}

/// One architecture at a time, for narrowing down a failure:
/// `PMETAL_GRAD_ARCH=llama4 cargo test --test adapter_gradients one_architecture -- --ignored`.
#[test]
#[ignore = "diagnostic; set PMETAL_GRAD_ARCH"]
fn one_architecture() {
    let want = std::env::var("PMETAL_GRAD_ARCH").expect("PMETAL_GRAD_ARCH");
    let (name, config_json) = all_cases()
        .into_iter()
        .find(|(name, _)| *name == want)
        .unwrap_or_else(|| panic!("no case named {want}"));
    let problems = check_architecture(name, config_json);
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
