//! Gradient checkpointing on a real architecture, end to end.
//!
//! `pmetal-bridge` proves the primitive gives the same gradients as a retained
//! backward, on shapes small enough to reason about. This proves the same thing
//! through the whole chain a training run uses: `AdaptedModel` ->
//! `DynamicModel` -> the architecture's own layer loop -> `checkpointed_layer`.
//!
//! The chain has a failure mode the primitive's own tests cannot see. Gradients
//! reach only the arrays handed to `checkpoint` as explicit inputs, so if the
//! layer loop rebuilds its forward from the module's own parameter copies
//! rather than the traced ones, every adapter gradient comes back zero. The
//! loss still matches, the run still completes, and the adapter never moves.

use std::collections::HashMap;
use std::rc::Rc;

use pmetal_bridge::compat::{Array, ModuleParametersExt};
use pmetal_bridge::inline_array::value_and_grad;
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, TrainableModel};
use pmetal_models::dispatcher::DynamicModel;

/// Two architectures whose trunks implement checkpointing, one per shape of
/// layer loop: llama branches on the cache, qwen3 threads it through the index.
const CASES: &[(&str, &str)] = &[
    (
        "llama",
        r#"{
            "model_type": "llama",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
    (
        "qwen3",
        r#"{
            "model_type": "qwen3",
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#,
    ),
];

const SEQ_LEN: i32 = 32;

fn build(config_json: &str) -> AdaptedModel {
    let config = LoraConfig {
        r: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    let model = DynamicModel::from_config(config_json).expect("architecture should build");
    AdaptedModel::attach(model, config).expect("adapters should attach")
}

/// One training step. Returns the loss and the adapter gradients keyed by
/// parameter path, so two runs can be compared name by name.
fn step(model: &mut AdaptedModel, checkpoint: bool) -> (f32, HashMap<Rc<str>, Array>) {
    if checkpoint {
        model.enable_gradient_checkpointing(1);
    } else {
        model.disable_gradient_checkpointing();
    }

    let mut names: Vec<Rc<str>> = model.flatten_trainable_params().into_keys().collect();
    names.sort();
    let live = model.flatten_trainable_params();
    let params: Vec<Array> = names.iter().map(|name| live[name].clone()).collect();
    drop(live);

    let tokens: Vec<i32> = (0..SEQ_LEN).map(|i| (i * 7 + 3) % 128).collect();
    let ids = Array::from_slice(&tokens, &[1, SEQ_LEN]);

    let (loss, grads) = value_and_grad(
        |arrays| {
            let restored: HashMap<Rc<str>, Array> = names
                .iter()
                .cloned()
                .zip(arrays[..names.len()].iter().cloned())
                .collect();
            model.set_lora_parameters(&restored);
            // Sum of logits: a scalar that depends on every adapter, and needs
            // no label tensor.
            TrainableModel::forward(model, &arrays[names.len()], None)
                .unwrap()
                .sum_all()
        },
        &params,
        std::slice::from_ref(&ids),
    );

    (loss.item_f32(), names.into_iter().zip(grads).collect())
}

#[test]
fn checkpointing_does_not_change_training() {
    for (name, config_json) in CASES {
        let mut model = build(config_json);
        assert!(
            model.supports_gradient_checkpointing(),
            "{name} reports no gradient-checkpointing support, so this case proves nothing"
        );

        // Same model, same step, twice. `step` writes the parameters it was
        // handed back into the adapters, so the second run starts where the
        // first did.
        let (retained_loss, retained_grads) = step(&mut model, false);
        let (ckpt_loss, ckpt_grads) = step(&mut model, true);

        assert!(
            (retained_loss - ckpt_loss).abs() / retained_loss.abs().max(1.0) < 1e-4,
            "{name}: checkpointed loss {ckpt_loss} differs from retained {retained_loss}"
        );
        assert_eq!(
            retained_grads.len(),
            ckpt_grads.len(),
            "{name}: checkpointing changed which parameters got gradients"
        );

        for (param, retained) in &retained_grads {
            let checkpointed = &ckpt_grads[param];
            let scale = retained.abs().max(None).item_f32().max(1.0);
            let diff = checkpointed.subtract(retained).abs().max(None).item_f32();
            assert!(
                diff / scale < 1e-4,
                "{name}: gradient for {param} differs by {diff} under checkpointing"
            );
        }

        let moved = ckpt_grads
            .values()
            .any(|grad| grad.abs().max(None).item_f32() > 0.0);
        assert!(
            moved,
            "{name}: every adapter gradient came back zero under checkpointing, so the \
             layer loop is rebuilding its forward from the module's own parameters \
             instead of the ones `checkpoint` is tracing"
        );
    }
}

#[test]
fn the_support_check_and_the_setter_agree() {
    // `supports_gradient_checkpointing` and `set_gradient_checkpointing` carry
    // separate match arms on `DynamicModel`, because callers have to ask before
    // enabling and enabling needs `&mut self`. This holds the two lists
    // against each other, including for architectures that support neither.
    let all_configs = CASES.iter().map(|(name, json)| (*name, *json));
    for (name, config_json) in all_configs {
        let mut model = DynamicModel::from_config(config_json).unwrap();
        assert_eq!(
            model.supports_gradient_checkpointing(),
            model.set_gradient_checkpointing(true),
            "{name}: the support check and the setter disagree"
        );
    }
}

#[test]
fn checkpointing_is_off_until_asked_for() {
    // Inference shares this trunk, and a recompute nobody asked for is pure
    // cost there.
    for (name, config_json) in CASES {
        let model = build(config_json);
        assert!(
            !model.model().is_gradient_checkpointing(),
            "{name} started up with gradient checkpointing already on"
        );
    }
}
