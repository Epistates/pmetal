//! The `/v1/systemone` request and response bodies.
//!
//! Request: `{"model": str, "state": any, "questions": {id: question}}`, plus
//! optional `images` / `videos`. Response:
//! `{"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens": 0}}`,
//! with answers in request order:
//!
//! * `noul`: `{"type", "noul"}`, the probability of `true`.
//! * `choice`: `{"type", "choice", "confidence", "probabilities"}`, options in
//!   the order the request lists them; ties go to the first.
//! * `score`: `{"type", "score", "confidence", "legend", "probabilities"}`, where
//!   `score` is the expected level.
//!
//! Probabilities are a per-question softmax in f32, and every number is rounded
//! to four places the way Python's `round(x, 4)` rounds a float.

use serde_json::{Map, Number, Value, json};

use super::DecisionError;
use super::encode::{EncodedRecord, QuestionType, is_truthy, score_levels};

/// Check a request body, with the reference's error messages.
pub fn validate_request(request: &Value) -> Result<(), DecisionError> {
    let request = request
        .as_object()
        .ok_or_else(|| DecisionError::Request("request body must be a JSON object".into()))?;
    if !matches!(request.get("model"), Some(Value::String(_))) || !request.contains_key("state") {
        return Err(DecisionError::Request(
            "model and state are required".into(),
        ));
    }
    let questions = match request.get("questions") {
        Some(Value::Object(questions)) if !questions.is_empty() => questions,
        _ => {
            return Err(DecisionError::Request(
                "at least one question is required".into(),
            ));
        }
    };
    for (question_id, question) in questions {
        let question = question.as_object().ok_or_else(|| {
            DecisionError::Request(format!("{question_id}: question must be an object"))
        })?;
        let question_type = question
            .get("type")
            .and_then(Value::as_str)
            .and_then(QuestionType::parse)
            .ok_or_else(|| {
                DecisionError::Request(format!(
                    "{question_id}: type must be noul, choice, or score"
                ))
            })?;
        if question_type != QuestionType::Noul && !question.get("criteria").is_some_and(is_truthy) {
            return Err(DecisionError::Request(format!(
                "{question_id}: criteria must not be empty"
            )));
        }
    }
    Ok(())
}

/// Softmax in f32, as `logits.float().softmax(-1)` computes it.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / total).collect()
}

/// Python's `round(x, 4)` on a float: the exact binary value rounded to four
/// decimals, ties to even.
fn round4(value: f64) -> Value {
    let rounded: f64 = format!("{value:.4}")
        .parse()
        .expect("a formatted float parses");
    Number::from_f64(rounded).map_or(Value::Null, Value::Number)
}

/// One question's answer from its option probabilities.
///
/// `probabilities` maps option id to probability.
pub fn answer(
    question_id: &str,
    question: &Map<String, Value>,
    probabilities: &Map<String, Value>,
) -> Result<Value, DecisionError> {
    let probability = |option: &str| -> Result<f64, DecisionError> {
        probabilities
            .get(option)
            .and_then(Value::as_f64)
            .ok_or_else(|| DecisionError::Request(format!("{question_id}: no option {option}")))
    };
    let question_type = question
        .get("type")
        .and_then(Value::as_str)
        .and_then(QuestionType::parse)
        .ok_or_else(|| {
            DecisionError::Request(format!(
                "{question_id}: type must be noul, choice, or score"
            ))
        })?;
    let criteria = question.get("criteria").unwrap_or(&Value::Null);
    match question_type {
        QuestionType::Noul => Ok(json!({"type": "noul", "noul": round4(probability("true")?)})),
        QuestionType::Choice => {
            let options = criteria.as_object().ok_or_else(|| {
                DecisionError::Request(format!(
                    "{question_id}: choice criteria must be an object of option descriptions"
                ))
            })?;
            let mut best: Option<(&String, f64)> = None;
            let mut rounded = Map::new();
            for option in options.keys() {
                let p = probability(option)?;
                if best.is_none_or(|(_, top)| p > top) {
                    best = Some((option, p));
                }
                rounded.insert(option.clone(), round4(p));
            }
            let (choice, confidence) = best.expect("validated criteria are not empty");
            Ok(json!({
                "type": "choice",
                "choice": choice,
                "confidence": round4(confidence),
                "probabilities": rounded,
            }))
        }
        QuestionType::Score => {
            let legend_values = score_levels(question_id, criteria)?;
            let mut score = 0.0f64;
            let mut confidence = f64::NEG_INFINITY;
            let mut legend = Map::new();
            let mut rounded = Map::new();
            for (index, description) in legend_values.into_iter().enumerate() {
                let level = index.to_string();
                let p = probability(&level)?;
                score += index as f64 * p;
                confidence = confidence.max(p);
                legend.insert(level.clone(), description);
                rounded.insert(level, round4(p));
            }
            Ok(json!({
                "type": "score",
                "score": round4(score),
                "confidence": round4(confidence),
                "legend": legend,
                "probabilities": rounded,
            }))
        }
    }
}

/// The response body for `request`, given the head's logits for its encoding.
pub fn response(
    request: &Value,
    encoded: &EncodedRecord,
    logits: &[Vec<f32>],
) -> Result<Value, DecisionError> {
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .ok_or_else(|| DecisionError::Request("at least one question is required".into()))?;
    let mut answers = Map::new();
    for (question, question_logits) in encoded.questions.iter().zip(logits) {
        let probabilities: Map<String, Value> = question
            .option_ids
            .iter()
            .zip(softmax(question_logits))
            .map(|(id, p)| (id.clone(), Value::from(f64::from(p))))
            .collect();
        let spec = questions
            .get(&question.question_id)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                DecisionError::Request(format!("{}: question is missing", question.question_id))
            })?;
        answers.insert(
            question.question_id.clone(),
            answer(&question.question_id, spec, &probabilities)?,
        );
    }
    Ok(json!({
        "model": request.get("model").cloned().unwrap_or(Value::Null),
        "answers": answers,
        "usage": {"input_tokens": encoded.input_ids.len(), "output_tokens": 0},
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs(pairs: &[(&str, f64)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::from(*v)))
            .collect()
    }

    #[test]
    fn validation_messages_match_the_reference() {
        let cases = [
            (
                json!({"state": 1, "questions": {}}),
                "model and state are required",
            ),
            (
                json!({"model": "m", "questions": {}}),
                "model and state are required",
            ),
            (
                json!({"model": "m", "state": null}),
                "at least one question is required",
            ),
            (
                json!({"model": "m", "state": null, "questions": {}}),
                "at least one question is required",
            ),
            (
                json!({"model": "m", "state": 1, "questions": {"q": {"type": "bool"}}}),
                "q: type must be noul, choice, or score",
            ),
            (
                json!({"model": "m", "state": 1, "questions": {"q": {"type": "choice", "criteria": {}}}}),
                "q: criteria must not be empty",
            ),
            (
                json!({"model": "m", "state": 1, "questions": {"q": {"type": "score"}}}),
                "q: criteria must not be empty",
            ),
        ];
        for (request, message) in cases {
            assert_eq!(validate_request(&request).unwrap_err().to_string(), message);
        }
        let ok = json!({"model": "m", "state": null, "questions": {"q": {"type": "noul"}}});
        assert!(validate_request(&ok).is_ok());
    }

    #[test]
    fn choice_keeps_request_order_and_the_first_maximum() {
        let question = json!({"type": "choice", "criteria": {"z": "Z", "a": "A", "m": "M"}});
        let answer = answer(
            "q",
            question.as_object().unwrap(),
            &probs(&[("a", 0.4), ("m", 0.2), ("z", 0.4)]),
        )
        .unwrap();
        assert_eq!(answer["choice"], "z");
        let keys: Vec<&String> = answer["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        assert_eq!(keys, ["z", "a", "m"]);
    }

    #[test]
    fn score_is_the_expected_level() {
        let question = json!({"type": "score", "criteria": ["low", "mid", "high"]});
        let answer = answer(
            "q",
            question.as_object().unwrap(),
            &probs(&[("0", 0.25), ("1", 0.25), ("2", 0.5)]),
        )
        .unwrap();
        assert_eq!(answer["score"], 1.25);
        assert_eq!(answer["confidence"], 0.5);
        assert_eq!(
            answer["legend"],
            json!({"0": "low", "1": "mid", "2": "high"})
        );
    }

    #[test]
    fn rounding_is_python_round() {
        assert_eq!(round4(0.123_45), json!(0.1235));
        assert_eq!(round4(0.000_05), json!(0.0001));
        assert_eq!(round4(0.999_99), json!(1.0));
        assert_eq!(round4(2.675e-5), json!(0.0));
    }
}
