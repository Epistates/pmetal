//! Real-weight parity for Clef-flash against the release's PyTorch reference.
//!
//! `clef_flash_reference.json` holds five `/v1/systemone` requests (the model
//! card's two examples, a French support message, an entailment pair, a
//! multiple-choice question) answered by the release's own `load_release_model`
//! and `systemone` in float32 on CPU, dumped by
//! `.strategy/parity/dump_clef_real_weight_reference.py`.
//!
//! pmetal runs the backbone in bf16 on the GPU and the head in f32, so the two
//! sides agree to bf16 noise, not bit for bit. The measured worst per-option
//! probability difference is 5.3e-3; the reference's own bf16 run lands 2.8e-3
//! from its float32 one. Token counts, answer order and every `choice` must
//! match exactly.
//!
//! `clef_flash_media_reference.json` holds three more, carrying media in the
//! JSON form `decision::media` defines (base64 PNGs, a video as frames plus
//! fps): one image, two images, and a six-frame clip, dumped by
//! `.strategy/parity/dump_clef_media_reference.py`, which hands the reference
//! the same pixels. The vision tower also runs in bf16, and media move further:
//! pmetal's worst is 9.2e-3, the reference's own bf16 run lands 1.23e-2 from
//! its float32 one (both on the clip's `is_moving`), so the bound is 2e-2.
//!
//! Needs the weights (~19 GB): set `PMETAL_CLEF_DIR` to a Clef-flash snapshot.
//! Skipped, loudly, without it.

use pmetal_models::decision::{DEFAULT_MAX_LENGTH, DecisionModel};
use serde_json::Value;

/// About twice the measured worst case (5.3e-3).
const MAX_PROBABILITY_DIFF: f64 = 1e-2;

/// About twice pmetal's measured worst case (9.2e-3) and 1.6 times the
/// reference's own bf16-to-float32 distance on the same requests (1.23e-2).
const MAX_MEDIA_PROBABILITY_DIFF: f64 = 2e-2;

#[test]
fn clef_flash_matches_reference() {
    check_fixture("clef_flash_reference.json", MAX_PROBABILITY_DIFF);
}

#[test]
fn clef_flash_media_matches_reference() {
    check_fixture(
        "clef_flash_media_reference.json",
        MAX_MEDIA_PROBABILITY_DIFF,
    );
}

fn check_fixture(name: &str, max_probability_diff: f64) {
    let Ok(dir) = std::env::var("PMETAL_CLEF_DIR") else {
        eprintln!(
            "SKIPPED: set PMETAL_CLEF_DIR to a Clef-flash snapshot to run real-weight parity"
        );
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut model = DecisionModel::load(&dir).expect("Clef-flash loads");

    let mut worst = 0.0f64;
    for (index, case) in fixture["results"].as_array().unwrap().iter().enumerate() {
        let request = &case["request"];
        let started = std::time::Instant::now();
        let response = model.systemone(request, DEFAULT_MAX_LENGTH).unwrap();
        eprintln!(
            "request {index}: {:.0} ms",
            started.elapsed().as_secs_f64() * 1e3
        );

        let expected = &case["response"];
        assert_eq!(
            response["usage"], expected["usage"],
            "request {index}: token count"
        );
        let answers = response["answers"].as_object().unwrap();
        let expected_answers = expected["answers"].as_object().unwrap();
        assert!(
            answers.keys().eq(expected_answers.keys()),
            "request {index}: answers in request order"
        );
        for (question, answer) in answers {
            let reference = &expected_answers[question];
            assert_eq!(answer["type"], reference["type"]);
            assert_eq!(
                answer["choice"], reference["choice"],
                "request {index} {question}"
            );
            let probabilities = &case["probabilities"][question];
            let ours: Vec<(String, f64)> = match answer.get("probabilities") {
                Some(Value::Object(map)) => map
                    .iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                    .collect(),
                _ => vec![("true".into(), answer["noul"].as_f64().unwrap())],
            };
            for (option, p) in ours {
                let diff = (p - probabilities[&option].as_f64().unwrap()).abs();
                worst = worst.max(diff);
                assert!(
                    diff < max_probability_diff,
                    "request {index} {question}/{option}: {p} vs reference {}",
                    probabilities[&option]
                );
            }
        }
    }
    eprintln!("{name}: worst per-option probability difference: {worst:.4}");
}
