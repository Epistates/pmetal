//! Numerical parity for the Clef joint schema head.
//!
//! The oracle is the release's own PyTorch `JointSchemaHead`
//! (`joint_schema_model.py`), run by `.strategy/parity/dump_clef_head_reference.py`
//! on a small seeded head: hidden 48, width 32, two routing layers, two decoder layers,
//! four heads. The fixture's record has a choice, a proposition and a score
//! question with spans of one to four tokens, and the scalars are set so each
//! matters: `prior_logit_scale = 5.0` is above the `log(100)` clamp,
//! `joint_logit_scale = 1.3`, `residual_gate = 0.4`.
//!
//! The head is checked in f32 against f32 torch, so the only allowed difference
//! is reduction order.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_bridge::compat::Dtype;
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::decision::{
    EncodedQuestion, EncodedRecord, JointHeadConfig, JointSchemaHead, QuestionType,
};
use serde::Deserialize;

const REFERENCE: &str = "clef_head_reference.safetensors";
const WEIGHTS: &str = "clef_head_weights.safetensors";

/// Set from the measured worst case (5.7e-6 on logits of magnitude 20) with
/// headroom. A tanh GELU in place of the exact one misses by 7e-5, so this
/// must stay below that.
const TOL: Tolerance = Tolerance::new(1e-5, 1e-6);

#[derive(Deserialize)]
struct Meta {
    config: JointHeadConfig,
    questions: Vec<MetaQuestion>,
}

#[derive(Deserialize)]
struct MetaQuestion {
    question_id: String,
    question_type: i32,
    question_span: (usize, usize),
    option_spans: Vec<(usize, usize)>,
    option_ids: Vec<String>,
}

fn question_type(index: i32) -> QuestionType {
    match index {
        0 => QuestionType::Noul,
        1 => QuestionType::Choice,
        2 => QuestionType::Score,
        other => panic!("question type {other}"),
    }
}

#[test]
fn joint_schema_head_matches_reference() {
    let meta_path = fixture_path(&format!("{REFERENCE}.meta.json"));
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    let reference = load_shard(&fixture_path(REFERENCE));
    let weights = load_shard(&fixture_path(WEIGHTS));

    let mut head = JointSchemaHead::new(meta.config.clone()).unwrap();
    head.load_weights(weights, Dtype::Float32)
        .expect("the torch state_dict loads strictly");

    let input_ids = ref_tensor(&reference, "input_ids").clone();
    input_ids.eval();
    let record = EncodedRecord {
        input_ids: input_ids
            .as_slice::<i32>()
            .iter()
            .map(|&id| id as u32)
            .collect(),
        questions: meta
            .questions
            .iter()
            .map(|q| EncodedQuestion {
                question_id: q.question_id.clone(),
                question_type: question_type(q.question_type),
                question_span: q.question_span,
                option_spans: q.option_spans.clone(),
                option_ids: q.option_ids.clone(),
            })
            .collect(),
        record_id: "fixture".into(),
    };

    let logits = head
        .forward(
            ref_tensor(&reference, "hidden_states"),
            &record,
            ref_tensor(&reference, "output_embedding"),
        )
        .expect("head forward");
    pmetal_bridge::check_last_error().expect("no bridge op threw");

    let reports: Vec<ParityReport> = logits
        .iter()
        .enumerate()
        .map(|(index, rust)| {
            let name = format!("logits_{index}");
            ParityReport::compute(&name, rust, ref_tensor(&reference, &name), TOL)
        })
        .collect();
    print_report_table(&reports);
    for report in &reports {
        assert!(
            report.passed(),
            "{} diverges from the reference",
            report.name
        );
    }
}

#[test]
fn a_mismatched_checkpoint_is_refused() {
    let meta_path = fixture_path(&format!("{REFERENCE}.meta.json"));
    let meta: Meta = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();

    let mut weights = load_shard(&fixture_path(WEIGHTS));
    weights.remove("residual_gate");
    let mut head = JointSchemaHead::new(meta.config.clone()).unwrap();
    let err = head.load_weights(weights, Dtype::Float32).unwrap_err();
    assert!(err.to_string().contains("residual_gate"), "{err}");

    let mut weights = load_shard(&fixture_path(WEIGHTS));
    let extra = weights["field_norm.bias"].clone();
    weights.insert("field_norm.extra".into(), extra);
    let mut head = JointSchemaHead::new(meta.config).unwrap();
    let err = head.load_weights(weights, Dtype::Float32).unwrap_err();
    assert!(err.to_string().contains("field_norm.extra"), "{err}");
}
