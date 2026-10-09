//! Holds the **training** forward pass against the **inference** forward pass.
//!
//! `pmetal-lora` does not wrap `pmetal-models`; it re-derives every
//! architecture from the same `Config` struct. That means each architecture has
//! two independent implementations, and a fix landed in one does not reach the
//! other. Nothing in the suite asked whether they still agree.
//!
//! A freshly constructed LoRA model initialises `lora_b` to zeros, so the
//! adapter contributes exactly nothing and the wrapped model *is* the base
//! model. Feed both paths the same checkpoint and the logits must match to
//! floating-point noise. Anything above that is the training path computing a
//! different function from the one that will serve the adapter.
//!
//! ## How a case is staged
//!
//! Weights are generated rather than committed, so this test needs no fixture
//! and carries no upstream model license:
//!
//! 1. [`DynamicModel::from_config`] builds the architecture from the case's
//!    `config.json` and random-initialises it. This is the *sized* constructor,
//!    not the placeholder one `load` uses.
//! 2. `flatten_params` yields that random init keyed by parameter path, which
//!    (after [`checkpoint_key`]) is the checkpoint format.
//! 3. Write it as `model.safetensors`. Both paths now load one identical set of
//!    weights through their own production loaders.
//!
//! Step 1 is what keeps this architecture-agnostic: no per-architecture weight
//! generator, and a checkpoint that is correct by construction.

mod common;

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use common::{ArchCase, SEQ_LEN, cases, config_int, input_ids, stage};
use pmetal_bridge::compat::{Array, ModuleParametersExt};
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, DynamicLoraModel, TrainableModel};
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, max_abs_diff, max_abs_value, print_report_table,
};
use pmetal_models::dispatcher::DynamicModel;

/// Both paths run the same ops in f32 over the same weights, so the only
/// legitimate difference is op-ordering noise (a fused SDPA on one side and an
/// explicit softmax on the other reassociate the same sums). The relative part
/// carries the gate; `atol` only keeps near-zero logits from tripping it.
const TOLERANCE: Tolerance = Tolerance::new(1e-3, 1e-3);

/// Run one architecture and return its report, or the reason it could not run.
fn run_case(case: &ArchCase) -> Result<ParityReport, String> {
    let dir = stage(case)?;
    let result = compare(case, &dir);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn compare(case: &ArchCase, dir: &Path) -> Result<ParityReport, String> {
    let vocab = config_int(case, "vocab_size")?;
    let ids = input_ids(vocab);

    let mut reference = DynamicModel::load(dir).map_err(|e| format!("reference load: {e}"))?;
    let ref_logits = reference
        .forward(&ids, None)
        .map_err(|e| format!("reference forward: {e}"))?;

    // Two identically-degenerate outputs would sail through any diff. Insist
    // the reference is a real distribution over the configured vocab before
    // comparing anything to it.
    if ref_logits.shape() != [1, SEQ_LEN, vocab] {
        return Err(format!(
            "reference logits are {:?}, expected [1, {SEQ_LEN}, {vocab}] — the staged \
             checkpoint did not populate this architecture",
            ref_logits.shape()
        ));
    }
    if max_abs_value(&ref_logits) < 1e-3 {
        return Err("reference logits are all but zero, so the comparison is vacuous".to_string());
    }

    // Rank 8 with the stock zero-init on `lora_b`: adapters are present and
    // trainable, and contribute nothing until the first optimizer step.
    let lora_config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };
    let mut lora = DynamicLoraModel::from_pretrained(dir, lora_config)
        .map_err(|e| format!("lora load: {e}"))?;
    let lora_logits =
        TrainableModel::forward(&mut lora, &ids, None).map_err(|e| format!("lora forward: {e}"))?;

    if ref_logits.shape() != lora_logits.shape() {
        return Err(format!(
            "shape mismatch: inference {:?} vs training {:?}",
            ref_logits.shape(),
            lora_logits.shape()
        ));
    }

    let report = ParityReport::compute(case.name, &lora_logits, &ref_logits, TOLERANCE);
    Ok(report)
}

/// Every architecture reachable from both dispatchers must agree.
///
/// Reported as a table so a regression names the architecture that broke rather
/// than failing on whichever case happens to run first.
///
/// Architectures carrying a [`ArchCase::known_divergence`] are held to the
/// opposite assertion: they *must still fail*. That keeps the gate honest in
/// both directions. A new divergence breaks the build, and repairing a listed
/// one also breaks the build until its entry is deleted, so the list cannot rot
/// into a permanent suppression.
#[test]
fn lora_forward_matches_base_forward() {
    let mut reports = Vec::new();
    let mut failures = Vec::new();
    let mut fixed = Vec::new();

    for case in cases().into_iter().filter(|case| case.stages) {
        match run_case(&case) {
            Ok(report) => {
                let detail = format!(
                    "max_abs={:.3e} mean_abs={:.3e} cos={:.6}",
                    report.max_abs_diff, report.mean_abs_diff, report.cosine_similarity
                );
                match (report.passed(), case.known_divergence) {
                    (false, None) => failures.push(format!("{}: {detail}", case.name)),
                    (true, Some(note)) => fixed.push(format!(
                        "{}: now agrees ({detail}). Delete its `known_divergence`: {note}",
                        case.name
                    )),
                    _ => {}
                }
                reports.push(report);
            }
            Err(reason) => failures.push(format!("{}: {reason}", case.name)),
        }
    }

    print_report_table(&reports);
    for case in cases() {
        if let Some(note) = case.known_divergence {
            println!("known divergence — {}: {note}", case.name);
        }
    }

    assert!(
        fixed.is_empty(),
        "an architecture listed as divergent now agrees:\n  {}",
        fixed.join("\n  ")
    );
    assert!(
        failures.is_empty(),
        "the training forward pass diverges from the inference forward pass:\n  {}",
        failures.join("\n  ")
    );
}

/// The zero-init contract the test above rests on.
///
/// If `lora_b` ever stopped being zero-initialised, `lora_forward_matches_base_forward`
/// would fail for a reason that has nothing to do with architecture drift. This
/// separates the two so the diagnosis is immediate.
#[test]
fn fresh_adapters_contribute_nothing() {
    let case = &cases()[0];
    let dir = stage(case).expect("stage llama");

    let lora_config = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };
    let lora = DynamicLoraModel::from_pretrained(&dir, lora_config).expect("load lora");
    let params = lora.lora_parameters();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!params.is_empty(), "no adapter parameters were created");

    let mut b_count = 0;
    for (name, value) in &params {
        if name.ends_with("lora_b") {
            b_count += 1;
            assert_eq!(
                max_abs_value(value),
                0.0,
                "{name} is not zero-initialised, so a fresh adapter perturbs the base model"
            );
        }
    }
    assert!(
        b_count > 0,
        "no lora_b parameters found among {} params",
        params.len()
    );
}

/// Guard against a tolerance that passes because both sides produce garbage.
///
/// A model whose logits are all zero, or whose argmax is constant, would sail
/// through a diff-based comparison while telling us nothing.
#[test]
fn reference_forward_is_non_degenerate() {
    let case = &cases()[0];
    let dir = stage(case).expect("stage llama");
    let mut model = DynamicModel::load(&dir).expect("load reference");
    let logits = model.forward(&input_ids(256), None).expect("forward");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        max_abs_value(&logits) > 1e-3,
        "reference logits are all but zero, so the parity comparison is vacuous"
    );
    let argmax = argmax_last_axis(&logits);
    assert!(
        argmax.iter().any(|&t| t != argmax[0]),
        "reference argmax is constant across every position"
    );
}

/// Distinct modules must occupy distinct parameter paths.
///
/// `ModuleParamRef::extend` merges a child's map at the top level rather than
/// nesting it, so extending with two children that both expose `weight` makes
/// the second silently replace the first. `DeepSeekLoraModel` did that with
/// `embed_tokens` and `norm`, and the flattened tree lost the embedding
/// entirely -- invisible to training, which reaches adapters through
/// `lora_parameters`, but wrong for anything keyed on parameter paths.
#[test]
fn parameter_paths_do_not_collide() {
    for case in cases() {
        let Ok(dir) = stage(&case) else { continue };
        let lora_config = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            ..Default::default()
        };
        let model = DynamicLoraModel::from_pretrained(&dir, lora_config);
        let _ = std::fs::remove_dir_all(&dir);
        let Ok(model) = model else { continue };

        let params = model.flatten_params();
        // A path that is a strict prefix of another means one module was merged
        // into a parent's map instead of nested under its own name, which is the
        // shape that produces a collision.
        for name in params.keys() {
            assert!(
                !name.is_empty(),
                "{}: a parameter flattened to an empty path",
                case.name
            );
            assert!(
                !params
                    .keys()
                    .any(|other| other.as_ref() != name.as_ref()
                        && other.ends_with(&format!(".{name}"))),
                "{}: `{name}` is also the tail of another path, so two modules \
                 share a name at different depths",
                case.name
            );
        }
    }
}

/// Every projection the walker reaches must sit at the same path its weight
/// occupies in the parameter tree.
///
/// This is the invariant adapter targeting depends on. `target_modules` names
/// projections the way a checkpoint does (`q_proj`, `gate_proj`), so a walker
/// that agreed with the struct layout but not with the parameter tree would
/// attach adapters to the wrong layers, or to none.
#[test]
fn the_linear_walker_agrees_with_the_parameter_tree() {
    use pmetal_bridge::compat::VisitLinears;

    for case in cases() {
        let Ok(mut model) = DynamicModel::from_config(case.config_json) else {
            continue;
        };

        let weight_paths: std::collections::HashSet<String> = model
            .flatten_params()
            .keys()
            .filter_map(|k| k.strip_suffix(".weight").map(str::to_string))
            .collect();

        let mut visited: Vec<String> = Vec::new();
        model.visit_linears_mut("", &mut |path, _| visited.push(path.to_string()));

        assert!(
            !visited.is_empty(),
            "{}: the walker reached no projections at all",
            case.name
        );

        let mut seen = std::collections::HashSet::new();
        for path in &visited {
            assert!(
                seen.insert(path.clone()),
                "{}: two projections share the path `{path}`",
                case.name
            );
            assert!(
                weight_paths.contains(path),
                "{}: walker reached `{path}`, which has no `{path}.weight` in the \
                 parameter tree",
                case.name
            );
        }
    }
}

// ─── The generic path ────────────────────────────────────────────────────────
//
// `AdaptedModel` attaches adapters to the model the inference path builds,
// rather than re-deriving the architecture. The tests below are the ones the
// per-architecture path could never pass by construction: there is one forward
// pass, so agreement is structural rather than something each architecture has
// to be talked into.

fn lora_config() -> LoraConfig {
    LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    }
}

/// Attaching adapters must not change what the model computes.
///
/// Note what is *not* needed here: a staged checkpoint, or a second model to
/// compare against. The adapted model is the base model, so this compares it to
/// itself before and after.
#[test]
fn attaching_adapters_changes_nothing() {
    let mut checked = 0;
    for case in cases() {
        let Ok(mut base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let vocab = config_int(&case, "vocab_size").expect("vocab_size");
        let ids = input_ids(vocab);

        let before = base.forward(&ids, None).expect("base forward");
        let mut adapted = AdaptedModel::attach(base, lora_config()).expect("attach");
        let after = adapted.forward(&ids, None).expect("adapted forward");

        assert_eq!(
            max_abs_diff(&before, &after),
            0.0,
            "{}: attaching adapters perturbed the model",
            case.name
        );
        assert!(
            !adapted.adapted_projections().is_empty(),
            "{}: no projection was adapted",
            case.name
        );
        checked += 1;
    }
    assert!(checked >= 15, "only {checked} architectures were exercised");
}

/// Every adapter changes what the model computes once it is trained.
///
/// The fresh-adapter tests above can't see an adapter the forward pass never
/// reads, since a zero `B` contributes nothing either way. Nemotron-H's
/// projections and Qwen 3.5's MLPs multiplied by their weights directly, so a
/// LoRA run trained adapters that the model ignored.
#[test]
fn every_adapter_reaches_the_output() {
    let silent = common::silent_adapters(|_| {});
    assert!(
        silent.is_empty(),
        "adapters that never reach the output:\n  {}",
        silent.join("\n  ")
    );
}

/// Every architecture takes a finite training step and comes out the other
/// side with a lower loss.
///
/// DeepSeek's and Nemotron-H's routers handed the expert gather indices that
/// still carried a gradient path, which MLX refuses to differentiate: every
/// LoRA step on either produced a NaN loss.
#[test]
fn every_architecture_trains() {
    let lora = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        ..Default::default()
    };
    let mut failures = Vec::new();
    for case in cases() {
        let Ok(base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let mut model = AdaptedModel::attach(base, lora.clone()).expect("attach");
        let ids = input_ids(config_int(&case, "vocab_size").expect("vocab"));
        let mut optimizer = pmetal_bridge::compat::optimizers::AdamW::new(1e-2, 0.0);
        let losses: Vec<f32> = (0..8)
            .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
            .collect();
        let _ = pmetal_bridge::check_last_error();
        if !losses.iter().all(|l| l.is_finite()) || losses[7] >= losses[0] {
            failures.push(format!("{}: {losses:?}", case.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Cut cross-entropy needs the LM head matrix, and falls back to standard CE
/// when it cannot get one. The fallback is silent and costs the whole
/// `[batch, seq, vocab]` logits tensor, so a naming drift in one architecture
/// would show up as a memory regression rather than a failure.
///
/// Its shape has to be `[vocab, hidden]`, tied or not, because that is the
/// orientation the loss multiplies against.
#[test]
fn every_architecture_hands_over_its_lm_head() {
    let mut checked = 0;
    for case in cases() {
        let Ok(base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let vocab = config_int(&case, "vocab_size").expect("vocab_size");
        let hidden = config_int(&case, "hidden_size").expect("hidden_size");

        let adapted = AdaptedModel::attach(base, lora_config()).expect("attach");
        let head = TrainableModel::lm_head(&adapted)
            .unwrap_or_else(|| panic!("{}: no LM head, so CCE falls back", case.name));

        assert_eq!(
            head.weight.shape(),
            &[vocab, hidden],
            "{}: LM head is not [vocab, hidden]",
            case.name
        );
        checked += 1;
    }
    assert!(checked >= 15, "only {checked} architectures were exercised");
}

/// Merging folds the adapters in and leaves a model that computes the same
/// thing, which is what `pmetal fuse` produces.
#[test]
fn merging_preserves_the_adapted_output() {
    let case = &cases()[0];
    let base = DynamicModel::from_config(case.config_json).expect("build");
    let mut adapted = AdaptedModel::attach(base, lora_config()).expect("attach");

    // Give the adapters something to contribute, as a training step would.
    let trained: HashMap<Rc<str>, Array> = adapted
        .lora_parameters()
        .into_iter()
        .map(|(k, v)| {
            let value = if k.ends_with("lora_b") {
                pmetal_bridge::compat::random::uniform_range(
                    -0.02,
                    0.02,
                    v.shape(),
                    pmetal_bridge::compat::Dtype::Float32,
                )
            } else {
                v
            };
            (k, value)
        })
        .collect();
    adapted.set_lora_parameters(&trained);

    let ids = input_ids(config_int(case, "vocab_size").expect("vocab"));
    let before = adapted.forward(&ids, None).expect("adapted forward");

    adapted.merge();
    let merged = adapted.forward(&ids, None).expect("merged forward");
    let report = ParityReport::compute("merge", &merged, &before, TOLERANCE);
    assert!(
        report.passed(),
        "merging changed the output: max_abs={:.3e} cos={:.6}",
        report.max_abs_diff,
        report.cosine_similarity
    );

    // Merging is reversible, so a run can merge for an evaluation pass and then
    // carry on training.
    adapted.unmerge();
    let unmerged = adapted.forward(&ids, None).expect("unmerged forward");
    let report = ParityReport::compute("unmerge", &unmerged, &before, TOLERANCE);
    assert!(
        report.passed(),
        "unmerging did not restore the model: max_abs={:.3e} cos={:.6}",
        report.max_abs_diff,
        report.cosine_similarity
    );

    // Fusing is the one-way form `pmetal fuse` produces.
    adapted.fuse();
    assert!(adapted.adapted_projections().is_empty());
    assert!(
        adapted.lora_parameters().is_empty(),
        "fuse left adapter tensors in the parameter tree"
    );
    let fused = adapted.forward(&ids, None).expect("fused forward");
    let report = ParityReport::compute("fuse", &fused, &before, TOLERANCE);
    assert!(
        report.passed(),
        "fusing changed the output: max_abs={:.3e} cos={:.6}",
        report.max_abs_diff,
        report.cosine_similarity
    );
}

/// Adapters survive a trip through a safetensors file.
#[test]
fn adapters_round_trip_through_a_file() {
    let case = &cases()[0];
    let base = DynamicModel::from_config(case.config_json).expect("build");
    let mut adapted = AdaptedModel::attach(base, lora_config()).expect("attach");

    let seeded: HashMap<Rc<str>, Array> = adapted
        .lora_parameters()
        .into_iter()
        .map(|(k, v)| {
            let value = pmetal_bridge::compat::random::uniform_range(
                -0.1,
                0.1,
                v.shape(),
                pmetal_bridge::compat::Dtype::Float32,
            );
            (k, value)
        })
        .collect();
    adapted.set_lora_parameters(&seeded);
    let written = adapted.lora_parameters();

    let dir = std::env::temp_dir().join(format!("pmetal_adapter_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("adapters.safetensors");
    adapted.save_lora_weights(&path).expect("save");

    // Re-attach from scratch, then load: the adapters must come back identical.
    let fresh = DynamicModel::from_config(case.config_json).expect("build");
    let mut reloaded = AdaptedModel::attach(fresh, lora_config()).expect("attach");
    reloaded.load_lora_weights(&path).expect("load");
    let _ = std::fs::remove_dir_all(&dir);

    let recovered = reloaded.lora_parameters();
    assert_eq!(recovered.len(), written.len(), "adapter count changed");
    for (key, value) in &written {
        let got = recovered
            .get(key)
            .unwrap_or_else(|| panic!("`{key}` missing after reload"));
        assert_eq!(max_abs_diff(got, value), 0.0, "`{key}` came back different");
    }
}

/// `target_modules` selects which projections are adapted, by the name a
/// checkpoint uses.
#[test]
fn targeting_selects_projections_by_name() {
    let case = &cases()[0];
    let base = DynamicModel::from_config(case.config_json).expect("build");
    let config = LoraConfig {
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        ..lora_config()
    };
    let adapted = AdaptedModel::attach(base, config).expect("attach");

    let names: Vec<&str> = adapted
        .adapted_projections()
        .iter()
        .map(|p| p.rsplit('.').next().unwrap_or(p))
        .collect();
    assert!(!names.is_empty(), "nothing was adapted");
    assert!(
        names.iter().all(|n| *n == "q_proj" || *n == "v_proj"),
        "adapted something outside target_modules: {names:?}"
    );
    assert!(names.contains(&"q_proj") && names.contains(&"v_proj"));
}

/// A target that matches nothing is a mistake worth reporting, not a model that
/// silently trains zero parameters.
#[test]
fn an_unmatched_target_is_an_error() {
    let case = &cases()[0];
    let base = DynamicModel::from_config(case.config_json).expect("build");
    let config = LoraConfig {
        target_modules: vec!["not_a_projection".to_string()],
        ..lora_config()
    };
    assert!(AdaptedModel::attach(base, config).is_err());
}

/// Only the adapters are trainable, across the whole model.
///
/// `Linear` reports this per layer and the recursion carries it up, so nothing
/// in between has to know which projections were targeted. If this regressed, a
/// LoRA run would quietly hand every base weight to the optimizer -- a full
/// fine-tune wearing a LoRA config.
#[test]
fn only_adapters_are_trainable() {
    for case in cases() {
        let Ok(base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let adapted = AdaptedModel::attach(base, lora_config()).expect("attach");

        let trainable = adapted.flatten_trainable_params();

        assert!(
            !trainable.is_empty(),
            "{}: nothing is trainable at all",
            case.name
        );
        let offenders: Vec<&str> = trainable
            .keys()
            .map(|k| k.as_ref())
            .filter(|k| !k.contains("lora_"))
            .collect();
        assert!(
            offenders.is_empty(),
            "{}: {} non-adapter tensors are trainable, e.g. {:?}",
            case.name,
            offenders.len(),
            &offenders[..offenders.len().min(4)]
        );
        assert_eq!(
            trainable.len(),
            adapted.lora_parameters().len(),
            "{}: trainable set and adapter set disagree",
            case.name
        );
    }
}

/// The adapted model answers for the architecture it wraps.
///
/// The trainer asks these of whatever it is fine-tuning; delegating rather than
/// re-deriving is the point.
#[test]
fn the_adapted_model_reports_its_architecture() {
    for case in cases() {
        let Ok(base) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let expected = base.architecture();
        let mut adapted = AdaptedModel::attach(base, lora_config()).expect("attach");

        assert_eq!(adapted.architecture(), expected, "{}", case.name);
        assert!(
            TrainableModel::supports_kv_cache(&adapted),
            "{}: an adapted model should still support KV caching",
            case.name
        );
        assert!(
            TrainableModel::create_cache(&adapted, 64).is_some(),
            "{}: could not build a cache",
            case.name
        );

        // Hidden states must still be reachable, for cut cross-entropy.
        let ids = input_ids(config_int(&case, "vocab_size").expect("vocab"));
        assert!(
            adapted.forward_hidden(&ids, None).is_ok(),
            "{}: forward_hidden failed",
            case.name
        );
    }
}
