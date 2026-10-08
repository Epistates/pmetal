//! QLoRA on every architecture the dispatcher builds.
//!
//! QLoRA is two properties of the same `Linear`: a packed weight and an
//! adapter. That reaches an architecture only if its forward pass goes through
//! `Linear::forward` for every projection it packs; one that reads `.weight`
//! directly would multiply by packed `uint32` words, and would ignore an
//! adapter too. These tests hold each architecture to that.

mod common;

use common::{cases, config_int, input_ids};
use pmetal_bridge::compat::optimizers::AdamW;
use pmetal_bridge::compat::{Dtype, VisitLinears, random};
use pmetal_core::LoraConfig;
use pmetal_lora::{AdaptedModel, QLoraConfig, QLoraScheme, TrainableModel, quantize_base};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, max_abs_diff, print_report_table};
use pmetal_models::dispatcher::DynamicModel;

/// A packed model and the same model unpacked again carry the same rounded
/// weights, so the only difference left is the quantized matmul's own
/// accumulation order.
const TOLERANCE: Tolerance = Tolerance::new(2e-3, 2e-3);

const SCHEMES: [QLoraScheme; 3] = [QLoraScheme::Nf4, QLoraScheme::Fp4, QLoraScheme::Int8];

fn config(scheme: QLoraScheme) -> QLoraConfig {
    QLoraConfig {
        scheme,
        ..Default::default()
    }
}

/// Every packed projection computes what its unpacked weight computes: no
/// architecture reads a packed weight as if it were dense.
#[test]
fn a_packed_base_computes_what_its_unpacked_weights_compute() {
    let mut reports = Vec::new();
    let mut failures = Vec::new();
    // The same seed builds the same random weights twice.
    let build = |config_json: &str| {
        random::seed(7);
        DynamicModel::from_config(config_json).expect("build")
    };
    for case in cases() {
        let ids = input_ids(config_int(&case, "vocab_size").expect("vocab"));
        for scheme in SCHEMES {
            let run = std::panic::catch_unwind(|| {
                let mut packed = build(case.config_json);
                let report = quantize_base(&mut packed, &config(scheme)).expect("pack");
                assert!(report.packed > 0, "nothing was packed");

                let mut unpacked = build(case.config_json);
                quantize_base(&mut unpacked, &config(scheme)).expect("pack");
                unpacked.visit_linears_mut("", &mut |_, linear| linear.dequantize());

                let a = packed.forward(&ids, None).expect("packed forward");
                let b = unpacked.forward(&ids, None).expect("unpacked forward");
                pmetal_bridge::check_last_error().map_err(|e| e.to_string())?;
                Ok::<_, String>((a, b))
            });
            let (a, b) = match run {
                Ok(Ok(pair)) => pair,
                Ok(Err(e)) => {
                    failures.push(format!("{}/{scheme}: {e}", case.name));
                    continue;
                }
                Err(_) => {
                    let _ = pmetal_bridge::check_last_error();
                    failures.push(format!("{}/{scheme}: panicked", case.name));
                    continue;
                }
            };
            let report = ParityReport::compute(
                &format!("{}/{scheme}", case.name),
                &a.as_dtype(Dtype::Float32.as_i32()),
                &b.as_dtype(Dtype::Float32.as_i32()),
                TOLERANCE,
            );
            if !report.passed() {
                failures.push(format!(
                    "{}/{scheme}: max_abs={:.3e} cos={:.6}",
                    case.name, report.max_abs_diff, report.cosine_similarity
                ));
            }
            reports.push(report);
        }
    }
    print_report_table(&reports);
    assert!(
        failures.is_empty(),
        "a packed base computed something its own weights don't:\n  {}",
        failures.join("\n  ")
    );
}

/// What QLoRA packs and what it leaves: every projection the walker reaches
/// is packed, or kept for a reason, and the kept ones are the head, the
/// routers and the routed experts.
#[test]
fn every_projection_is_packed_or_kept_for_a_reason() {
    for case in cases() {
        let Ok(mut model) = DynamicModel::from_config(case.config_json) else {
            continue;
        };
        let mut total = 0;
        model.visit_linears_mut("", &mut |_, _| total += 1);
        let report = quantize_base(&mut model, &QLoraConfig::default()).expect("pack");
        assert_eq!(
            report.packed + report.kept.len(),
            total,
            "{}: a projection was neither packed nor kept",
            case.name
        );
        for (path, reason) in &report.kept {
            let name = path.rsplit('.').next().unwrap_or(path);
            let expected = match name {
                "lm_head" => reason.contains("LM head"),
                _ if path.contains(".experts.") => reason.contains("routed experts"),
                _ => reason.contains("router") || reason.contains("divide"),
            };
            assert!(expected, "{}: `{path}` kept: {reason}", case.name);
        }
        assert!(
            report.packed_bytes * 3 < report.dense_bytes,
            "{}: {} bytes packed from {}",
            case.name,
            report.packed_bytes,
            report.dense_bytes
        );
    }
}

/// An adapter on a packed projection changes the output: the architecture
/// routes the projection through `Linear::forward`, adapter included.
#[test]
fn an_adapter_on_a_packed_base_reaches_the_output() {
    let silent = common::silent_adapters(|base| {
        quantize_base(base, &QLoraConfig::default()).expect("pack");
    });
    assert!(
        silent.is_empty(),
        "adapters on a packed base that never reach the output:\n  {}",
        silent.join("\n  ")
    );
}

/// QLoRA trains on every architecture: a few steps on one sequence bring the
/// loss down with the base packed in each scheme, gradient checkpointing on as
/// `pmetal train` runs it, and the adapters it saves load onto the same model
/// unpacked, which is how `pmetal infer --lora` applies them.
#[test]
fn every_architecture_trains_with_a_packed_base() {
    let lora = LoraConfig {
        r: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: Vec::new(),
        ..Default::default()
    };
    let build = |config_json: &str| {
        random::seed(11);
        DynamicModel::from_config(config_json).expect("build")
    };
    let dir = std::env::temp_dir().join(format!("pmetal_qlora_train_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");

    let mut failures = Vec::new();
    for case in cases() {
        let ids = input_ids(config_int(&case, "vocab_size").expect("vocab"));
        for scheme in SCHEMES {
            let mut base = build(case.config_json);
            quantize_base(&mut base, &config(scheme)).expect("pack");
            let mut model = AdaptedModel::attach(base, lora.clone()).expect("attach");
            if model.supports_gradient_checkpointing() {
                model.enable_gradient_checkpointing(1);
            }
            let mut optimizer = AdamW::new(1e-2, 0.0);
            let losses: Vec<f32> = (0..12)
                .map(|_| common::train_step(&mut model, &mut optimizer, &ids))
                .collect();
            pmetal_bridge::check_last_error().expect("no bridge error");
            let (first, last) = (losses[0], losses[losses.len() - 1]);
            if !losses.iter().all(|l| l.is_finite()) || last > first - 0.25 {
                failures.push(format!("{}/{scheme}: {losses:?}", case.name));
                continue;
            }

            // Saved, then loaded onto the unpacked model.
            let path = dir.join(format!("{}-{scheme}.safetensors", case.name));
            model.save_lora_weights(&path).expect("save");
            let mut served =
                AdaptedModel::attach(build(case.config_json), lora.clone()).expect("attach");
            served.load_lora_weights(&path).expect("load");
            let (trained, loaded) = (model.lora_parameters(), served.lora_parameters());
            if trained.len() != loaded.len()
                || trained
                    .iter()
                    .any(|(k, v)| loaded.get(k).is_none_or(|l| max_abs_diff(l, v) != 0.0))
            {
                failures.push(format!(
                    "{}/{scheme}: the adapters didn't load onto the unpacked model",
                    case.name
                ));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        failures.is_empty(),
        "QLoRA didn't train:\n  {}",
        failures.join("\n  ")
    );
}
