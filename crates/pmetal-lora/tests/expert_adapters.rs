//! Adapters on the routed experts of a mixture-of-experts layer.
//!
//! The experts hold most of a MoE model's parameters, and attention-only (or
//! attention-plus-dense-MLP) LoRA leaves them alone. Each stacked expert
//! projection takes a per-expert adapter, sized so the rank a token sees is the
//! configured one. Gradients against finite differences, and a silent-adapter
//! check, run over these in `adapter_gradients.rs` and `base_parity.rs`; here
//! is what is particular to experts.

mod common;

use common::{cases, input_ids};
use pmetal_cache::test_utils::max_abs_diff;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, ModuleParametersExt, VisitLinears, random};
use std::collections::HashMap;
use std::rc::Rc;

/// Architectures whose routed experts are the shared stacked layout.
const STACKED: [&str; 3] = ["qwen3_moe", "deepseek", "nemotron_h"];

fn json(name: &str) -> &'static str {
    cases()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no `{name}` case"))
        .config_json
}

fn build(name: &str) -> DynamicModel {
    random::seed(61);
    DynamicModel::from_config(json(name)).expect("build")
}

fn lora(r: usize) -> LoraConfig {
    LoraConfig {
        r,
        alpha: 16.0,
        ..Default::default()
    }
}

fn drain() {
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

fn randomize(model: &mut AdaptedModel) {
    random::seed(9);
    let mut params: Vec<_> = model.lora_parameters().into_iter().collect();
    params.sort_by(|a, b| a.0.cmp(&b.0));
    let set: HashMap<Rc<str>, Array> = params
        .into_iter()
        .map(|(name, p)| {
            (
                name,
                random::uniform_range(-0.1, 0.1, p.shape(), pmetal_runtime::compat::Dtype::Float32),
            )
        })
        .collect();
    model.set_lora_parameters(&set);
}

/// The default adapts the routed experts, one adapter per expert, of rank
/// `r / top_k` (at least one).
#[test]
fn the_default_adapts_every_routed_expert_at_rank_over_top_k() {
    for name in STACKED {
        let mut base = build(name);
        let mut shapes = Vec::new();
        base.visit_experts_mut("", 1, &mut |path, top_k, projection| {
            shapes.push((path.to_string(), top_k, projection.stack_shape().unwrap()));
        });
        assert!(!shapes.is_empty(), "{name}: no stacked experts");

        let model = AdaptedModel::attach(base, lora(8)).expect("attach");
        let params = model.lora_parameters();
        for (path, top_k, (experts, out, input)) in shapes {
            let stem = path.strip_prefix("model.").unwrap_or(&path);
            let rank = (8 / top_k as i32).max(1);
            assert_eq!(
                params[format!("{stem}.lora_a").as_str()].shape(),
                &[experts, rank, input],
                "{name}: {path}"
            );
            assert_eq!(
                params[format!("{stem}.lora_b").as_str()].shape(),
                &[experts, out, rank],
                "{name}: {path}"
            );
        }
    }
}

/// A model with expert adapters starts out computing the base model, and
/// folding the adapters in (`merge`) or out (`fuse`) leaves the adapted
/// forward unchanged; `unmerge` puts the experts back.
#[test]
fn expert_adapters_start_neutral_and_merge_exactly() {
    for name in STACKED {
        let case = cases().into_iter().find(|c| c.name == name).unwrap();
        let ids = input_ids(common::config_int(&case, "vocab_size").unwrap());
        let want = build(name).forward(&ids, None).unwrap();

        let mut model = AdaptedModel::attach(build(name), lora(8)).unwrap();
        let fresh = model.forward(&ids, None).unwrap();
        assert!(
            max_abs_diff(&fresh, &want) < 1e-5,
            "{name}: fresh adapters moved the logits"
        );

        randomize(&mut model);
        let adapted = model.forward(&ids, None).unwrap();
        assert!(
            max_abs_diff(&adapted, &want) > 1e-3,
            "{name}: the adapters changed nothing"
        );

        model.merge().expect("merge");
        let merged = model.forward(&ids, None).unwrap();
        assert!(max_abs_diff(&merged, &adapted) < 5e-4, "{name}: merge");
        model.unmerge();
        let back = model.forward(&ids, None).unwrap();
        assert!(max_abs_diff(&back, &adapted) < 5e-4, "{name}: unmerge");

        model.fuse().expect("fuse");
        let fused = model.forward(&ids, None).unwrap();
        assert!(max_abs_diff(&fused, &adapted) < 5e-4, "{name}: fuse");
        assert!(
            model.lora_parameters().is_empty(),
            "{name}: fuse left adapters"
        );
        drain();
    }
}

/// Expert adapters survive a save and a load, and name no PEFT tensor they do
/// not mean: the factor pairs are `lora_expert_A` / `lora_expert_B`.
#[test]
fn expert_adapters_round_trip_through_a_file() {
    let name = "qwen3_moe";
    let ids = input_ids(256);
    let mut model = AdaptedModel::attach(build(name), lora(8)).unwrap();
    randomize(&mut model);
    let want = model.forward(&ids, None).unwrap();

    let dir = tempfile::tempdir().unwrap();
    model.save_adapter(dir.path(), None).unwrap();
    let names: Vec<String> =
        pmetal_lora::load_safetensors_map(dir.path().join("adapter_model.safetensors"))
            .unwrap()
            .into_keys()
            .collect();
    assert!(
        names
            .iter()
            .any(|n| n.ends_with("switch_mlp.gate_proj.lora_expert_A")),
        "{names:?}"
    );

    let mut loaded = AdaptedModel::attach(build(name), lora(8)).unwrap();
    loaded.load_lora_weights(dir.path()).unwrap();
    let got = loaded.forward(&ids, None).unwrap();
    assert_eq!(max_abs_diff(&got, &want), 0.0);

    // Another rank is refused, not reshaped.
    let mut wide = AdaptedModel::attach(build(name), lora(16)).unwrap();
    let err = wide.load_lora_weights(dir.path()).unwrap_err();
    assert!(err.to_string().contains("rank"), "{err}");
    drain();
}

/// Training reaches the experts: a few steps lower the loss, and the expert
/// adapters moved.
#[test]
fn the_experts_train() {
    let ids = input_ids(256);
    let mut model = AdaptedModel::attach(build("qwen3_moe"), lora(8)).unwrap();
    let before: HashMap<_, _> = model.lora_parameters();
    let mut optimizer = pmetal_runtime::compat::optimizers::AdamW::new(1e-2, 0.0);
    let losses: Vec<f32> = (0..10)
        .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
        .collect();
    drain();
    assert!(losses[9] < losses[0] - 0.05, "{losses:?}");
    let after = model.lora_parameters();
    let moved = after
        .iter()
        .filter(|(k, _)| k.contains("switch_mlp") && k.ends_with(".lora_b"))
        .filter(|(k, v)| max_abs_diff(v, &before[*k]) > 0.0)
        .count();
    assert!(moved >= 3, "only {moved} expert B matrices moved");
    let _ = model.flatten_trainable_params();
}

/// Naming only attention leaves the experts alone.
#[test]
fn explicit_attention_targets_leave_the_experts_out() {
    let config = LoraConfig {
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..lora(8)
    };
    let model = AdaptedModel::attach(build("qwen3_moe"), config).unwrap();
    assert!(
        model
            .adapted_projections()
            .iter()
            .all(|p| !p.contains("switch_mlp"))
    );
}
