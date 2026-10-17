//! Parameters that must never move under training stay out of the trainable
//! set.
//!
//! Some tensors a model carries are fixed by the architecture: Llama 4's QK
//! norm has no weight at all, a Cohere layer norm has no bias, a MoE router
//! picks experts by argmax, and a buffer such as DeepSeek's load-balancing
//! offset is updated by a rule of its own. Held as ordinary parameters they
//! reach the optimizer in a full fine-tune, and weight decay or a gradient
//! moves them off the identity the reference runs with. The model then
//! computes something no other framework does, and the checkpoint it saves
//! carries tensors the architecture does not have.

mod common;

use common::cases;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::ModuleParametersExt;

/// Whether a path names a router (or a gate that decides routing).
fn is_router_path(path: &str) -> bool {
    path.split('.')
        .any(|seg| matches!(seg, "router" | "gate" | "shared_expert_gate"))
}

/// A parameter the loader is allowed to leave at its constructed value is one
/// the model must not train: the checkpoint never supplies it, so its value is
/// part of the architecture.
#[test]
fn no_parameter_the_loader_leaves_at_its_init_is_trainable() {
    let mut checked = 0;
    for case in cases() {
        let Ok(model) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        checked += 1;
        let policy = model.architecture().load_policy();
        let trainable = model.flatten_trainable_params();
        for name in trainable.keys() {
            assert!(
                !policy.unloaded_params.iter().any(|e| e.covers(name)),
                "{}: `{name}` is left at its init by the loader and still trainable",
                case.name
            );
        }
    }
    assert!(checked >= 15, "only {checked} architectures built");
}

/// Llama 4's QK norm is a parameter-free L2 norm, and a Cohere layer norm has
/// no bias: neither exists as a tensor, so neither can be trained or saved.
#[test]
fn parameter_free_norms_have_no_parameters() {
    for (name, banned) in [
        (
            "llama4",
            ["self_attn.q_norm", "self_attn.k_norm"].as_slice(),
        ),
        ("cohere", ["layernorm.bias", "norm.bias"].as_slice()),
    ] {
        let case = cases()
            .into_iter()
            .find(|c| c.name == name)
            .expect("case exists");
        let model = DynamicModel::from_config(case.config_json).expect("builds");
        for key in model.flatten_params().keys() {
            for banned in banned {
                // Cohere's per-head q/k norms are real weights; only the
                // layer-norm bias and Llama 4's QK norm are fixed.
                if name == "cohere" && !key.ends_with(".bias") {
                    continue;
                }
                assert!(
                    !key.contains(banned),
                    "{name}: `{key}` should not be a parameter"
                );
            }
        }
    }
}

/// Routers and buffers are never handed to an optimizer.
#[test]
fn routers_and_buffers_are_not_trainable() {
    let mut routers_seen = 0;
    let mut buffers_seen = 0;
    for case in cases() {
        let Ok(model) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let all = model.flatten_params();
        let trainable = model.flatten_trainable_params();
        routers_seen += all.keys().filter(|k| is_router_path(k)).count();
        buffers_seen += all
            .keys()
            .filter(|k| k.ends_with("e_score_correction_bias"))
            .count();
        for key in trainable.keys() {
            assert!(
                !is_router_path(key),
                "{}: router parameter `{key}` is trainable",
                case.name
            );
            assert!(
                !key.ends_with("e_score_correction_bias"),
                "{}: buffer `{key}` is trainable",
                case.name
            );
        }
    }
    // Guards: the fixtures contain routers and a buffer to be filtered.
    assert!(routers_seen >= 5, "only {routers_seen} router parameters");
    assert!(buffers_seen >= 1, "no buffer in any fixture");
}
