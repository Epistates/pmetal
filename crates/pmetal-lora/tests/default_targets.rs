//! What a default `LoraConfig` adapts.
//!
//! Attention-only LoRA underperforms full fine-tuning even at equal parameter
//! count, while the MLP (or the MoE layer) holds most of the parameters. The
//! default therefore reaches every linear layer but the ones that must stay
//! exact, on every architecture, and every surface inherits it from
//! `LoraConfig::default()`.

mod common;

use std::collections::BTreeSet;

use common::cases;
use pmetal_core::LoraConfig;
use pmetal_lora::AdaptedModel;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::VisitLinears;

const ATTENTION: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

fn name_of(path: &str) -> &str {
    path.rsplit('.').next().unwrap_or(path)
}

/// Adapted paths of a default-config model built from `config_json`, and the
/// output layer's path.
fn adapted_by_default(config_json: &str) -> Option<(Vec<String>, Option<String>)> {
    let mut base = DynamicModel::from_config(config_json).ok()?;
    let head = base.output_projection_path();
    let model = AdaptedModel::attach(base, LoraConfig::default()).expect("attach");
    Some((model.adapted_projections().to_vec(), head))
}

/// On every architecture the default adapts a projection that is not one of
/// attention's four, and leaves out the head, routers and routed experts.
#[test]
fn the_default_reaches_past_attention_on_every_architecture() {
    let mut checked = 0;
    for case in cases() {
        let Some((adapted, head)) = adapted_by_default(case.config_json) else {
            continue;
        };
        checked += 1;
        // gpt-oss's experts are read by its own kernels, which take no adapter
        // yet; every other routed layer here is stacked experts that do.
        if case.name == "gpt_oss" {
            continue;
        }
        assert!(
            adapted.iter().any(|p| !ATTENTION.contains(&name_of(p))),
            "{}: the default adapted attention only: {adapted:?}",
            case.name
        );
        for path in &adapted {
            assert_ne!(Some(path), head.as_ref(), "{}: adapted the head", case.name);
            let name = name_of(path);
            assert!(
                !matches!(name, "gate" | "router" | "shared_expert_gate")
                    && !path.contains(".experts.")
                    && !path.ends_with(".router.layer")
                    && !path.ends_with(".router.proj"),
                "{}: adapted `{path}`, which must stay exact",
                case.name
            );
        }
    }
    assert!(checked >= 15, "only {checked} architectures built");
}

/// Dense, MoE and hybrid models name their MLP differently; the default finds
/// all of them.
#[test]
fn the_default_adapts_the_mlp_of_dense_moe_and_hybrid_models() {
    let wants: &[(&str, &[&str])] = &[
        ("llama", &["gate_proj", "up_proj", "down_proj"]),
        ("qwen3", &["gate_proj", "up_proj", "down_proj"]),
        // A MoE model's dense layers and shared expert.
        ("qwen3_next", &["gate_proj", "in_proj_qkv", "out_proj"]),
        ("nemotron_h", &["in_proj", "out_proj"]),
        ("granitemoehybrid", &["in_proj", "out_proj"]),
    ];
    for (case_name, names) in wants {
        let case = cases()
            .into_iter()
            .find(|c| c.name == *case_name)
            .unwrap_or_else(|| panic!("no `{case_name}` case"));
        let (adapted, _) = adapted_by_default(case.config_json)
            .unwrap_or_else(|| panic!("{case_name} does not build"));
        let present: BTreeSet<&str> = adapted.iter().map(|p| name_of(p)).collect();
        for name in *names {
            assert!(
                present.contains(name),
                "{case_name}: `{name}` not adapted by default; got {present:?}"
            );
        }
    }
}

/// The default and the spelling PEFT uses for it adapt the same set, and an
/// explicit list still means exactly that list.
#[test]
fn all_linear_equals_the_default_and_explicit_targets_still_win() {
    let case = cases()
        .into_iter()
        .find(|c| c.name == "llama")
        .expect("llama case");
    let build = || DynamicModel::from_config(case.config_json).expect("build");

    let default = AdaptedModel::attach(build(), LoraConfig::default()).expect("attach");
    let word = LoraConfig {
        target_modules: vec![LoraConfig::ALL_LINEAR.to_string()],
        ..Default::default()
    };
    let spelled = AdaptedModel::attach(build(), word).expect("attach");
    assert_eq!(default.adapted_projections(), spelled.adapted_projections());

    let explicit = LoraConfig {
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    let model = AdaptedModel::attach(build(), explicit).expect("attach");
    assert!(
        model
            .adapted_projections()
            .iter()
            .all(|p| matches!(name_of(p), "q_proj" | "v_proj"))
    );
    assert!(model.adapted_projections().len() < default.adapted_projections().len());

    // Guard: the model has MLP projections for the default to reach.
    let mut every = 0;
    build().visit_linears_mut("", &mut |_, _| every += 1);
    assert_eq!(every, default.adapted_projections().len() + 1, "the head");
}
