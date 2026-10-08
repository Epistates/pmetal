//! Every RoPE scaling type against transformers' `ROPE_INIT_FUNCTIONS`.
//!
//! `tests/fixtures/rope_scaling_reference.json` holds, per case, the config a
//! released model ships (or an edge case of the parser), the reach the init
//! function was called with, and the inverse frequencies and attention factor
//! transformers returned (`.strategy/parity/dump_rope_scaling_reference.py`).
//! The same config goes through [`Rotary::from_config`] here.

use pmetal_bridge::rope::{RopeConfig, Rotary};
use serde_json::Value;

fn reference() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rope_scaling_reference.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("fixture json")
}

#[test]
fn every_rope_type_matches_transformers() {
    let reference = reference();
    let cases = reference["cases"].as_array().expect("cases");
    assert!(cases.len() >= 30, "fixture has {} cases", cases.len());
    let mut seen = std::collections::BTreeSet::new();
    let mut worst = 0.0_f64;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let reach = case["reach"].as_i64().unwrap_or(0);
        let head_dim = case["head_dim"].as_i64().unwrap() as i32;
        let rotary = Rotary::from_config(
            head_dim,
            RopeConfig::from_json(&case["config"]),
            10_000.0,
            1.0,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            rotary.scaling.rope_type(),
            case["rope_type"].as_str().unwrap(),
            "{name}"
        );
        seen.insert(rotary.scaling.rope_type());

        let want: Vec<f64> = case["inv_freq"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got = rotary.inverse_frequencies(reach);
        assert_eq!(got.len(), want.len(), "{name}@{reach}: frequency count");
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let err = if w == 0.0 {
                (g as f64).abs()
            } else {
                ((g as f64) - w).abs() / w.abs()
            };
            worst = worst.max(err);
            assert!(
                err < 2e-6,
                "{name}@{reach}: inv_freq[{i}] = {g:e}, transformers {w:e} (rel {err:e})"
            );
        }

        let af = case["attention_factor"].as_f64().unwrap();
        assert!(
            (rotary.attention_factor() as f64 - af).abs() < 1e-6,
            "{name}: attention factor {} vs transformers {af}",
            rotary.attention_factor()
        );
    }
    eprintln!(
        "worst relative inv_freq error over {} cases: {worst:e}",
        cases.len()
    );
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        [
            "default",
            "dynamic",
            "linear",
            "llama3",
            "longrope",
            "proportional",
            "yarn"
        ],
        "every transformers rope_type is covered"
    );
}

/// The cases differ where they should: a fixture whose variants all agreed
/// would not pin the parser branch each one exists for.
#[test]
fn the_reference_cases_are_not_vacuous() {
    let reference = reference();
    let inv = |name: &str, reach: Option<i64>| -> Vec<f64> {
        reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name && c["reach"].as_i64() == reach)
            .unwrap_or_else(|| panic!("{name}@{reach:?}"))["inv_freq"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let differs =
        |a: &[f64], b: &[f64]| a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-3 * y.abs());
    assert!(differs(
        &inv("yarn_gpt_oss", None),
        &inv("yarn_gpt_oss_truncated", None)
    ));
    assert!(differs(
        &inv("yarn_custom_betas", None),
        &inv("yarn_zero_betas", None)
    ));
    assert!(differs(
        &inv("dynamic", Some(100)),
        &inv("dynamic", Some(64))
    ));
    assert!(differs(
        &inv("longrope_phi4_mini", Some(4097)),
        &inv("longrope_phi4_mini", Some(4096))
    ));
    assert!(differs(
        &inv("proportional_factor", None),
        &inv("proportional", None)
    ));
}
