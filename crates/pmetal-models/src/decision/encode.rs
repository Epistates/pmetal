//! Turning a decision record into one prompt, and remembering where each part of
//! the schema landed in it.
//!
//! The prompt is not rendered and then tokenized as a whole. Each piece (the
//! system turn, the state, every field header, instruction and option) is
//! tokenized on its own and the ids are concatenated, which is what lets the
//! encoder record a token span for each question's instruction and each option.
//! Tokenizing the joined string instead would merge tokens across piece
//! boundaries, shift every span, and give the head a different input than it was
//! trained on. [`encode_record`] reproduces the release's reference encoder
//! token for token; `tests/decision_encode_parity.rs` holds it to that.
//!
//! Non-string values are rendered the way Python's
//! `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`
//! renders them ([`python_json`]), since that string is what the model saw in
//! training: keys sorted, no whitespace, non-ASCII left as is, and floats in
//! Python's `repr` (`1250.0`, `1e-05`, `1e+16`).

use serde_json::{Map, Value};
use std::fmt::Write as _;

use super::DecisionError;

/// The system turn every record is encoded under.
pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
     Each answer must be exactly one of that field's allowed options.";

/// The reference encoder's default prompt budget, in tokens.
pub const DEFAULT_MAX_LENGTH: usize = 16384;

/// What a question asks for, and the index of its type embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionType {
    /// A proposition: the options are `true` and `false`.
    Noul,
    /// One of several named options.
    Choice,
    /// One of several ordered levels, indexed from 0.
    Score,
}

impl QuestionType {
    /// Parse the `type` field of a question.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "noul" => Some(Self::Noul),
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            _ => None,
        }
    }

    /// The spelling used in requests, responses and the prompt.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Noul => "noul",
            Self::Choice => "choice",
            Self::Score => "score",
        }
    }

    /// Row of the head's type embedding.
    pub fn index(self) -> i32 {
        match self {
            Self::Noul => 0,
            Self::Choice => 1,
            Self::Score => 2,
        }
    }
}

/// One question's place in the encoded prompt.
///
/// Spans are half-open `[start, end)` token ranges into
/// [`EncodedRecord::input_ids`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedQuestion {
    pub question_id: String,
    pub question_type: QuestionType,
    /// The rendered instruction.
    pub question_span: (usize, usize),
    /// One span per option, in [`option_ids`](Self::option_ids) order.
    pub option_spans: Vec<(usize, usize)>,
    /// Option ids in prompt order: `true`/`false` for a proposition, the keys
    /// sorted for a choice, `0..n` for a score.
    pub option_ids: Vec<String>,
}

/// A record encoded into one prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    /// In the order the record lists them.
    pub questions: Vec<EncodedQuestion>,
    /// The record's `id`, or `unknown`.
    pub record_id: String,
}

/// Prompt budget for [`encode_record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Longest prompt, in tokens. The state is truncated to fit; a schema that
    /// does not fit on its own is an error.
    pub max_length: usize,
    /// Keep at most this many state tokens, before `max_length` applies.
    pub max_state_tokens: Option<usize>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            max_length: DEFAULT_MAX_LENGTH,
            max_state_tokens: None,
        }
    }
}

/// A value as the prompt shows it: strings verbatim, everything else as compact
/// JSON with sorted keys.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_json(other),
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
///
/// One divergence is inherent to parsing with `serde_json`: an integer outside
/// the 64-bit range arrives as a float, where Python would keep every digit.
pub fn python_json(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => out.push_str(&python_float_repr(float)),
            _ => out.push_str(&number.to_string()),
        },
        Value::String(text) => write_json_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Python sorts `str` keys by code point, which is the byte order of
            // their UTF-8 encoding.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

/// Python's `ensure_ascii=False` string encoder: quote, backslash and the C0
/// controls are escaped, nothing else is.
fn write_json_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest digits that round-trip, positional for
/// decimal exponents in `[-4, 16)`, scientific (`1e-05`, `1.5e+16`) otherwise,
/// and always a `.0` on a whole number written positionally.
pub fn python_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .to_string();
    }
    // Rust's `{:e}` is also the shortest round-trip representation, so only
    // the layout differs.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` always writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa
        .trim_start_matches('-')
        .chars()
        .filter(|c| *c != '.')
        .collect();

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let whole = exponent as usize + 1;
            if digits.len() <= whole {
                out.push_str(&digits);
                out.push_str(&"0".repeat(whole - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..whole]);
                out.push('.');
                out.push_str(&digits[whole..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        let _ = write!(out, "e{sign}{:02}", exponent.abs());
    }
    out
}

/// Python truthiness of a JSON value.
pub(crate) fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|n| n != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// A question's options as `(id, description)`, in prompt order.
///
/// * `noul`: `true` then `false`, with default descriptions that the question's
///   `criteria` object may override.
/// * `choice`: the `criteria` object's entries, sorted by id.
/// * `score`: the `criteria` list's entries, with ids `"0"`, `"1"`, ….
///
/// A `null` description leaves the option described by its id alone.
pub fn question_options(
    question_id: &str,
    question: &Map<String, Value>,
) -> Result<Vec<(String, Value)>, DecisionError> {
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
        QuestionType::Noul => {
            let mut descriptions = Map::new();
            descriptions.insert(
                "true".into(),
                "The proposition is true or the answer is yes.".into(),
            );
            descriptions.insert(
                "false".into(),
                "The proposition is false or the answer is no.".into(),
            );
            if is_truthy(criteria) {
                let overrides = criteria.as_object().ok_or_else(|| {
                    DecisionError::Request(format!(
                        "{question_id}: noul criteria must be an object with true and false"
                    ))
                })?;
                for (key, value) in overrides {
                    descriptions.insert(key.clone(), value.clone());
                }
            }
            Ok(["true", "false"]
                .into_iter()
                .map(|key| (key.to_string(), descriptions[key].clone()))
                .collect())
        }
        QuestionType::Choice => {
            let options = criteria.as_object().ok_or_else(|| {
                DecisionError::Request(format!(
                    "{question_id}: choice criteria must be an object of option descriptions"
                ))
            })?;
            let mut options: Vec<(String, Value)> = options
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            options.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(options)
        }
        QuestionType::Score => Ok(score_levels(question_id, criteria)?
            .into_iter()
            .enumerate()
            .map(|(index, value)| (index.to_string(), value))
            .collect()),
    }
}

/// A score question's level descriptions, in order.
///
/// The reference enumerates `criteria`, so an object contributes its keys and a
/// string its characters; both are accepted for the same reason.
pub(crate) fn score_levels(
    question_id: &str,
    criteria: &Value,
) -> Result<Vec<Value>, DecisionError> {
    match criteria {
        Value::Array(items) => Ok(items.clone()),
        Value::Object(map) => Ok(map.keys().map(|key| Value::String(key.clone())).collect()),
        Value::String(text) => Ok(text.chars().map(|c| Value::String(c.to_string())).collect()),
        _ => Err(DecisionError::Request(format!(
            "{question_id}: score criteria must be a list of level descriptions"
        ))),
    }
}

/// `tokenizer` as the `tokenize` argument of [`encode_record`]: no special
/// tokens added, since the prompt spells its own.
pub fn tokenize_with(
    tokenizer: &pmetal_data::Tokenizer,
) -> impl FnMut(&str) -> Result<Vec<u32>, DecisionError> + '_ {
    |text| {
        tokenizer
            .encode(text)
            .map_err(|e| DecisionError::Tokenizer(e.to_string()))
    }
}

/// Encode `record` into one prompt with the instruction and option spans of
/// every question.
///
/// `tokenize` must tokenize without adding special tokens; the prompt spells
/// its own.
pub fn encode_record<F>(
    mut tokenize: F,
    record: &Value,
    options: EncodeOptions,
) -> Result<EncodedRecord, DecisionError>
where
    F: FnMut(&str) -> Result<Vec<u32>, DecisionError>,
{
    let record = record
        .as_object()
        .ok_or_else(|| DecisionError::Request("record must be a JSON object".into()))?;
    for key in ["images", "videos"] {
        if record.get(key).is_some_and(is_truthy) {
            return Err(DecisionError::Unsupported(format!(
                "{key} are not supported yet: pmetal has no Qwen3.5 vision encoder, so this \
                 server answers text-only records"
            )));
        }
    }
    let questions = record
        .get("questions")
        .and_then(Value::as_object)
        .ok_or_else(|| DecisionError::Request("at least one question is required".into()))?;
    let state = record
        .get("state")
        .ok_or_else(|| DecisionError::Request("model and state are required".into()))?;

    let mut schema_ids = tokenize("\n\nSCHEMA FIELDS:\n")?;
    let mut encoded_questions = Vec::with_capacity(questions.len());
    for (question_index, (question_id, question)) in questions.iter().enumerate() {
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
        schema_ids.extend(tokenize(&format!(
            "\nFIELD {}\nID: {question_id}\nTYPE: {}\nINSTRUCTION: ",
            question_index + 1,
            question_type.as_str()
        ))?);
        let question_start = schema_ids.len();
        let instruction = match question.get("instructions") {
            None | Some(Value::Null) => Value::String(question_id.clone()),
            Some(Value::String(text)) if text.is_empty() => Value::String(question_id.clone()),
            Some(value) => value.clone(),
        };
        schema_ids.extend(tokenize(&render(&instruction))?);
        let question_end = schema_ids.len();
        schema_ids.extend(tokenize("\nALLOWED OPTIONS:\n")?);

        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (option_index, (option_id, description)) in question_options(question_id, question)?
            .into_iter()
            .enumerate()
        {
            schema_ids.extend(tokenize(&format!("OPTION {}: ", option_index + 1))?);
            let option_start = schema_ids.len();
            let mut semantics = Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if !description.is_null() {
                semantics.insert("description".into(), description);
            }
            schema_ids.extend(tokenize(&render(&Value::Object(semantics)))?);
            option_spans.push((option_start, schema_ids.len()));
            option_ids.push(option_id);
            schema_ids.extend(tokenize("\n")?);
        }
        schema_ids.extend(tokenize("END FIELD\n")?);
        encoded_questions.push(EncodedQuestion {
            question_id: question_id.clone(),
            question_type,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let prefix_ids = tokenize(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let suffix_ids = tokenize(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let mut state_ids = tokenize(&render(state))?;
    if let Some(limit) = options.max_state_tokens {
        state_ids.truncate(limit);
    }
    let fixed_length = prefix_ids.len() + schema_ids.len() + suffix_ids.len();
    if fixed_length > options.max_length {
        return Err(DecisionError::Request(format!(
            "schema requires {fixed_length} tokens before state; maximum is {}",
            options.max_length
        )));
    }
    state_ids.truncate(options.max_length - fixed_length);

    let schema_offset = prefix_ids.len() + state_ids.len();
    let shift = |(start, end): (usize, usize)| (start + schema_offset, end + schema_offset);
    for question in &mut encoded_questions {
        question.question_span = shift(question.question_span);
        for span in &mut question.option_spans {
            *span = shift(*span);
        }
    }

    let mut input_ids = prefix_ids;
    input_ids.extend(state_ids);
    input_ids.extend(schema_ids);
    input_ids.extend(suffix_ids);
    if input_ids.is_empty() || encoded_questions.is_empty() {
        return Err(DecisionError::Request(
            "record produced no model input or questions".into(),
        ));
    }
    let record_id = match record.get("id") {
        None => "unknown".to_string(),
        Some(Value::String(id)) => id.clone(),
        Some(other) => python_str(other),
    };
    Ok(EncodedRecord {
        input_ids,
        questions: encoded_questions,
        record_id,
    })
}

/// Python's `str()` of a decoded JSON scalar, for the record id.
fn python_str(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(text) => text.clone(),
        other => python_json(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (value, expected) in [
            (1250.0, "1250.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (123456789012345.0, "123456789012345.0"),
            (1234567890123456.0, "1234567890123456.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.2345e-7, "1.2345e-07"),
            (1e300, "1e+300"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (1.0 / 3.0, "0.3333333333333333"),
        ] {
            assert_eq!(python_float_repr(value), expected, "repr({value:e})");
        }
    }

    #[test]
    fn json_is_compact_sorted_and_unescaped() {
        let value = json!({"b": [1, 2.0, null, true], "a": {"z": "é\"\n\u{1}", "y": -3}});
        assert_eq!(
            python_json(&value),
            "{\"a\":{\"y\":-3,\"z\":\"é\\\"\\n\\u0001\"},\"b\":[1,2.0,null,true]}"
        );
        assert_eq!(render(&json!("plain \"text\"")), "plain \"text\"");
    }

    /// One token per character, so spans are easy to read off.
    fn char_tokens(text: &str) -> Result<Vec<u32>, DecisionError> {
        Ok(text.chars().map(|c| c as u32).collect())
    }

    #[test]
    fn spans_cover_the_rendered_instruction_and_options() {
        let record = json!({
            "state": {"x": 1},
            "questions": {
                "pick": {"type": "choice", "criteria": {"b": "Bee", "a": null}},
                "ok": {"type": "noul", "instructions": "Is it?"},
            }
        });
        let encoded = encode_record(char_tokens, &record, EncodeOptions::default()).unwrap();
        let text: String = encoded
            .input_ids
            .iter()
            .map(|&id| char::from_u32(id).unwrap())
            .collect();
        let slice = |(start, end): (usize, usize)| {
            text.chars()
                .skip(start)
                .take(end - start)
                .collect::<String>()
        };

        let pick = &encoded.questions[0];
        assert_eq!(pick.option_ids, ["a", "b"]);
        assert_eq!(slice(pick.question_span), "pick");
        assert_eq!(slice(pick.option_spans[0]), "{\"option_id\":\"a\"}");
        assert_eq!(
            slice(pick.option_spans[1]),
            "{\"description\":\"Bee\",\"option_id\":\"b\"}"
        );
        let ok = &encoded.questions[1];
        assert_eq!(ok.question_type, QuestionType::Noul);
        assert_eq!(ok.option_ids, ["true", "false"]);
        assert_eq!(slice(ok.question_span), "Is it?");
        assert!(text.contains("STATE:\n{\"x\":1}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: pick\n"));
        assert!(text.ends_with("JOINT SCHEMA DECISIONS:"));
    }

    #[test]
    fn state_is_truncated_and_an_oversized_schema_is_refused() {
        let record = json!({"state": "abcdefghij", "questions": {"q": {"type": "noul"}}});
        let full = encode_record(char_tokens, &record, EncodeOptions::default()).unwrap();
        let fixed = full.input_ids.len() - 10;

        let capped = EncodeOptions {
            max_length: fixed + 4,
            max_state_tokens: None,
        };
        let short = encode_record(char_tokens, &record, capped).unwrap();
        assert_eq!(short.input_ids.len(), fixed + 4);
        assert_eq!(
            short.questions[0].question_span.0 + 6,
            full.questions[0].question_span.0
        );

        let state_cap = EncodeOptions {
            max_state_tokens: Some(3),
            ..EncodeOptions::default()
        };
        let short = encode_record(char_tokens, &record, state_cap).unwrap();
        assert_eq!(short.input_ids.len(), fixed + 3);

        let too_small = EncodeOptions {
            max_length: fixed - 1,
            max_state_tokens: None,
        };
        let err = encode_record(char_tokens, &record, too_small).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "schema requires {fixed} tokens before state; maximum is {}",
                fixed - 1
            )
        );
    }

    #[test]
    fn media_is_refused_rather_than_ignored() {
        let record =
            json!({"state": "x", "images": ["a.png"], "questions": {"q": {"type": "noul"}}});
        assert!(matches!(
            encode_record(char_tokens, &record, EncodeOptions::default()),
            Err(DecisionError::Unsupported(_))
        ));
        let empty = json!({"state": "x", "images": [], "questions": {"q": {"type": "noul"}}});
        assert!(encode_record(char_tokens, &empty, EncodeOptions::default()).is_ok());
    }
}
