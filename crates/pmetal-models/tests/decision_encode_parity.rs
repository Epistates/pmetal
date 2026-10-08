//! Encoding parity for Clef decision records.
//!
//! The oracle is the release's own `encode_record` and `render`
//! (`joint_schema_model.py`) with the release tokenizer, dumped by
//! `.strategy/parity/dump_clef_encoding_reference.py` into
//! `clef_encoding_reference.json`.
//!
//! * `render` cases need no tokenizer and always run: key order, Python float
//!   repr (300 random bit patterns among them), and string escaping.
//! * Record cases compare token ids, spans, option ids and error messages, and
//!   need the release tokenizer: set `PMETAL_CLEF_DIR` to a Clef or Clef-flash
//!   directory. Without it they are skipped, loudly.

use pmetal_models::decision::{EncodeOptions, encode::tokenize_with, encode_record, render};
use serde_json::Value;

fn fixture() -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/clef_encoding_reference.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn render_matches_python_json_dumps() {
    let fixture = fixture();
    let cases = fixture["render"].as_array().unwrap();
    assert!(cases.len() > 300);
    let mut failures = Vec::new();
    for case in cases {
        let (value, expected) = (&case[0], case[1].as_str().unwrap());
        let got = render(value);
        if got != expected {
            failures.push(format!("{value}: got {got:?}, expected {expected:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn records_encode_token_for_token() {
    let Ok(dir) = std::env::var("PMETAL_CLEF_DIR") else {
        eprintln!(
            "SKIPPED: set PMETAL_CLEF_DIR to a Clef release directory to run encoding parity"
        );
        return;
    };
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).expect("release tokenizer");
    let fixture = fixture();
    let records = fixture["records"].as_array().unwrap();
    for (index, case) in records.iter().enumerate() {
        let record: Value = serde_json::from_str(case["record_json"].as_str().unwrap()).unwrap();
        let mut options = EncodeOptions::default();
        if let Some(max_length) = case["options"]["max_length"].as_u64() {
            options.max_length = max_length as usize;
        }
        options.max_state_tokens = case["options"]["max_state_tokens"]
            .as_u64()
            .map(|n| n as usize);
        let result = encode_record(tokenize_with(&tokenizer), &record, options);

        if let Some(error) = case.get("error") {
            let got = result.expect_err("the reference refused this record");
            assert_eq!(got.to_string(), error.as_str().unwrap(), "record {index}");
            continue;
        }
        let encoded = result.unwrap_or_else(|e| panic!("record {index}: {e}"));
        let expected_ids: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(encoded.input_ids, expected_ids, "record {index}: input_ids");
        assert_eq!(
            encoded.record_id,
            case["record_id"].as_str().unwrap(),
            "record {index}"
        );
        let expected_questions = case["questions"].as_array().unwrap();
        assert_eq!(
            encoded.questions.len(),
            expected_questions.len(),
            "record {index}"
        );
        for (got, expected) in encoded.questions.iter().zip(expected_questions) {
            let span = |v: &Value| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            };
            assert_eq!(got.question_id, expected["question_id"].as_str().unwrap());
            assert_eq!(
                i64::from(got.question_type.index()),
                expected["question_type"].as_i64().unwrap(),
                "record {index} {}",
                got.question_id
            );
            assert_eq!(
                got.question_span,
                span(&expected["question_span"]),
                "record {index} {}",
                got.question_id
            );
            let option_spans: Vec<(usize, usize)> = expected["option_spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(span)
                .collect();
            assert_eq!(
                got.option_spans, option_spans,
                "record {index} {}",
                got.question_id
            );
            let option_ids: Vec<&str> = expected["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            assert_eq!(
                got.option_ids, option_ids,
                "record {index} {}",
                got.question_id
            );
        }
    }
    eprintln!("{} records encoded identically", records.len());
}
