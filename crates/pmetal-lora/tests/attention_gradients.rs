//! Attention gradients at the sequence lengths and in the mode a training run
//! uses.
//!
//! The training loops run every gradient step inside `with_training_mode`.
//! Under that flag the Qwen3.5 adapters used to hand Q/K/V to the Metal
//! FlashAttention kernel from 2048 tokens on, which copies them into Metal
//! buffers and builds its output from one. Autograd sees that output as a
//! constant, so with 128-wide heads every adapter on `q_proj`, `k_proj` and
//! `v_proj` trained on a zero gradient, and with Qwen3.5's real 256-wide
//! heads, which the kernel has no variant for, the step failed instead.
//! Attention called without a mask could also land on that kernel through
//! the timed backend choice in `fused_sdpa`, whichever mode it ran in.
//!
//! Each case here runs one training step in that mode at 2048 tokens and holds
//! the adapter gradients against a reference forward that cannot take the
//! Metal path, then checks a few entries against finite differences of the
//! loss itself.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{Array, Dtype, random};
use pmetal_bridge::inline_array::value_and_grad;
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, QLoraConfig, TrainableModel, quantize_base};
use pmetal_mlx::kernels::with_training_mode;
use pmetal_models::dispatcher::DynamicModel;

const SEQ_LEN: i32 = 2048;
const VOCAB: i32 = 64;
/// Positions whose logits enter the loss. Few, so the loss stays O(1) and a
/// finite difference resolves its derivative in f32.
const SCORED_TAIL: i32 = 8;

const LLAMA: &str = r#"{
    "model_type": "llama",
    "vocab_size": 64, "hidden_size": 128, "intermediate_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 2,
    "num_key_value_heads": 1, "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

/// Qwen3.5's layout (three GDN layers, then full attention) at its real
/// 256-wide heads.
fn qwen3_5_config(head_dim: i32) -> String {
    format!(
        r#"{{
        "model_type": "qwen3_next",
        "vocab_size": 64, "hidden_size": 128, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": {head_dim},
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2,
        "linear_key_head_dim": 32, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "num_experts": 0, "num_experts_per_tok": 0,
        "moe_intermediate_size": 32, "shared_expert_intermediate_size": 128,
        "partial_rotary_factor": 0.25, "tie_word_embeddings": false
    }}"#
    )
}

fn lora_config() -> LoraConfig {
    LoraConfig {
        r: 4,
        alpha: 8.0,
        target_modules: vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into(),
        ],
        ..Default::default()
    }
}

fn adapted(config_json: &str) -> AdaptedModel {
    let model = DynamicModel::from_config(config_json).expect("architecture should build");
    AdaptedModel::attach(model, lora_config()).expect("adapters should attach")
}

/// QLoRA: the base packed as NF4 (the default scheme), then adapters on top.
fn adapted_qlora(config_json: &str) -> AdaptedModel {
    let mut model = DynamicModel::from_config(config_json).expect("architecture should build");
    let packed = quantize_base(&mut model, &QLoraConfig::default()).expect("base should pack");
    assert!(packed.packed > 0, "nothing was packed");
    AdaptedModel::attach(model, lora_config()).expect("adapters should attach")
}

fn drain() {
    pmetal_bridge::check_last_error().expect("a bridge op threw");
}

/// Give every adapter (B included, which starts at zero) a nonzero value, so
/// both factors of every adapter have a gradient.
fn randomize_adapters(model: &mut impl TrainableModel) {
    random::seed(11);
    let params: HashMap<Rc<str>, Array> = model
        .lora_parameters()
        .into_iter()
        .map(|(name, p)| {
            let r = random::uniform_range(-0.05, 0.05, p.shape(), Dtype::Float32)
                .as_dtype(p.dtype().as_i32());
            (name, r)
        })
        .collect();
    model.set_lora_parameters(&params);
}

fn inputs() -> (Array, Array, Array) {
    let tokens: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 7 + 3) % VOCAB).collect();
    let ids = Array::from_slice(&tokens, &[1, SEQ_LEN]);
    let mut w = vec![0.0_f32; (SEQ_LEN * VOCAB) as usize];
    for (i, x) in w
        .iter_mut()
        .enumerate()
        .skip(((SEQ_LEN - SCORED_TAIL) * VOCAB) as usize)
    {
        *x = ((i * 37 % 101) as f32 / 101.0) - 0.5;
    }
    let weights = Array::from_f32_slice(&w, &[1, SEQ_LEN, VOCAB]);
    let causal = pmetal_models::architectures::utils::create_causal_mask(SEQ_LEN).unwrap();
    (ids, weights, causal)
}

fn loss_of(logits: &Array, weights: &Array) -> Array {
    logits
        .as_dtype(Dtype::Float32.as_i32())
        .multiply(weights)
        .sum_all()
}

/// How the attention layers are reached in a step.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// What the trainer does: training mode on, no mask from the caller.
    Training,
    /// Reference: an explicit causal mask array, training mode off.
    ReferenceMasked,
}

/// One step: loss and adapter gradients keyed by parameter path.
fn step(
    model: &mut impl TrainableModel,
    mode: Mode,
    names: &[Rc<str>],
) -> (f32, HashMap<Rc<str>, Array>) {
    let live = model.lora_parameters();
    let params: Vec<Array> = names.iter().map(|n| live[n].clone()).collect();
    let (ids, weights, causal) = inputs();

    let mut run = || {
        let (loss, grads) = value_and_grad(
            |arrays| {
                let restored: HashMap<Rc<str>, Array> = names
                    .iter()
                    .cloned()
                    .zip(arrays[..names.len()].iter().cloned())
                    .collect();
                model.set_lora_parameters(&restored);
                let mask = matches!(mode, Mode::ReferenceMasked).then_some(&causal);
                let logits = model.forward(&ids, mask).unwrap();
                loss_of(&logits, &weights)
            },
            &params,
            &[],
        );
        loss.eval();
        for g in &grads {
            g.eval();
        }
        Ok((loss.item_f32(), grads))
    };
    let (loss, grads) = match mode {
        Mode::Training => with_training_mode(run).unwrap(),
        _ => run().unwrap(),
    };
    drain();
    (loss, names.iter().cloned().zip(grads).collect())
}

fn sorted_names(model: &impl TrainableModel) -> Vec<Rc<str>> {
    let mut names: Vec<Rc<str>> = model.lora_parameters().into_keys().collect();
    names.sort();
    names
}

fn max_abs(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32.as_i32())
        .abs()
        .max(None)
        .item_f32()
}

fn assert_same_gradients(
    case: &str,
    (loss_a, grads_a): &(f32, HashMap<Rc<str>, Array>),
    (loss_b, grads_b): &(f32, HashMap<Rc<str>, Array>),
) {
    assert!(
        (loss_a - loss_b).abs() <= 1e-4 * loss_b.abs().max(1.0),
        "{case}: training loss {loss_a} vs reference {loss_b}"
    );
    for (name, reference) in grads_b {
        let got = &grads_a[name];
        let scale = max_abs(reference);
        let diff = max_abs(&got.subtract(reference));
        assert!(
            diff <= 1e-3 * scale.max(1e-6),
            "{case}: gradient of {name} differs by {diff} (reference scale {scale}; \
             training max {})",
            max_abs(got)
        );
    }
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        let moved = grads_a
            .iter()
            .filter(|(n, _)| n.contains(proj))
            .any(|(_, g)| max_abs(g) > 0.0);
        assert!(moved, "{case}: no {proj} adapter received a gradient");
    }
}

/// Central finite differences on the entries of a few adapter tensors where
/// the autograd gradient is largest.
fn assert_finite_differences(
    case: &str,
    model: &mut impl TrainableModel,
    grads: &HashMap<Rc<str>, Array>,
) {
    let (ids, weights, _) = inputs();
    let base = model.lora_parameters();
    for proj in ["q_proj", "k_proj", "v_proj"] {
        let name = grads
            .keys()
            .filter(|n| n.contains(proj))
            .max_by(|a, b| max_abs(&grads[*a]).total_cmp(&max_abs(&grads[*b])))
            .unwrap()
            .clone();
        let mut g = grads[&name].as_dtype(Dtype::Float32.as_i32());
        let n = g.size();
        let gv = g.to_f32_vec(n).unwrap();
        let (idx, &want) = gv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();

        let mut p = base[&name].as_dtype(Dtype::Float32.as_i32());
        let pv = p.to_f32_vec(n).unwrap();
        let h = 1e-2_f32;
        let mut loss_at = |delta: f32| {
            let mut v = pv.clone();
            v[idx] += delta;
            let mut params = base.clone();
            params.insert(
                name.clone(),
                Array::from_f32_slice(&v, p.shape()).as_dtype(base[&name].dtype().as_i32()),
            );
            model.set_lora_parameters(&params);
            let loss = with_training_mode(|| {
                Ok(loss_of(&model.forward(&ids, None).unwrap(), &weights).item_f32())
            })
            .unwrap();
            drain();
            loss
        };
        let fd = (loss_at(h) - loss_at(-h)) / (2.0 * h);
        model.set_lora_parameters(&base);
        assert!(
            (fd - want).abs() <= 2e-2 * want.abs() + 1e-4,
            "{case}: {name}[{idx}] autograd {want} vs finite difference {fd}"
        );
    }
}

fn check(case: &str, model: &mut impl TrainableModel, reference: Mode) {
    randomize_adapters(model);
    let names = sorted_names(model);
    let training = step(model, Mode::Training, &names);
    let reference = step(model, reference, &names);
    assert_same_gradients(case, &training, &reference);
    assert_finite_differences(case, model, &training.1);
}

#[test]
fn llama_attention_adapters_train_in_training_mode() {
    let mut model = adapted(LLAMA);
    check("llama d=64", &mut model, Mode::ReferenceMasked);
}

#[test]
fn qwen3_5_attention_adapters_train_at_2048_tokens() {
    for head_dim in [128, 256] {
        let mut model = adapted(&qwen3_5_config(head_dim));
        check(
            &format!("qwen3.5 LoRA d={head_dim}"),
            &mut model,
            Mode::ReferenceMasked,
        );
    }
}

#[test]
fn qwen3_5_qlora_attention_adapters_train_at_2048_tokens() {
    for head_dim in [128, 256] {
        let mut model = adapted_qlora(&qwen3_5_config(head_dim));
        check(
            &format!("qwen3.5 QLoRA d={head_dim}"),
            &mut model,
            Mode::ReferenceMasked,
        );
    }
}
