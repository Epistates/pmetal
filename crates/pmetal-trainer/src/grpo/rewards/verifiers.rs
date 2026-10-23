//! Verifiers over a completion's text: a pattern it must match, a JSON
//! document it must be (against a schema, if one is given).
//!
//! Each scores `1` for a completion that passes and `0` for one that does not.
//! A failed check is a judgement, not an error: these rewards are total over
//! text. Weight them with the rest of the combination.

use regex::Regex;
use serde_json::Value;

use super::{GrpoError, GrpoResult, RewardFunction};
use pmetal_runtime::compat::Array;

/// `1` for a completion the regular expression matches anywhere in, else `0`.
///
/// The pattern is a [`regex`] crate pattern (no look-around or
/// backreferences); anchor it with `^`/`$` to match the whole completion, and
/// use `(?s)` to let `.` cross lines.
#[derive(Debug, Clone)]
pub struct RegexReward {
    pattern: Regex,
}

impl RegexReward {
    /// A reward for `pattern`.
    ///
    /// # Errors
    ///
    /// [`GrpoError::Config`] if the pattern does not compile.
    pub fn new(pattern: &str) -> GrpoResult<Self> {
        Regex::new(pattern)
            .map(|pattern| Self { pattern })
            .map_err(|e| GrpoError::Config(format!("the reward pattern `{pattern}`: {e}")))
    }
}

impl RewardFunction for RegexReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        Ok(completions
            .iter()
            .map(|completion| f64::from(self.pattern.is_match(completion)))
            .collect())
    }

    fn name(&self) -> &str {
        "regex"
    }
}

/// `1` for a completion that is a JSON document (and, with a schema, one that
/// is valid against it), else `0`.
///
/// The document is the whole completion, or the contents of its first fenced
/// code block (` ```json ... ``` `), which models write JSON in. Text around
/// the document makes the completion fail, so the reward teaches a model to
/// answer with the document alone.
pub struct JsonReward {
    schema: Option<jsonschema::Validator>,
}

impl std::fmt::Debug for JsonReward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonReward")
            .field("schema", &self.schema.is_some())
            .finish()
    }
}

impl JsonReward {
    /// A reward for any valid JSON.
    pub fn any() -> Self {
        Self { schema: None }
    }

    /// A reward for JSON that is valid against `schema`, a JSON Schema
    /// document (draft 4 through 2020-12, picked by its `$schema`, 2020-12
    /// without one). References to other documents are not fetched.
    ///
    /// # Errors
    ///
    /// [`GrpoError::Config`] if `schema` is not a valid schema.
    pub fn with_schema(schema: &Value) -> GrpoResult<Self> {
        let validator = jsonschema::validator_for(schema)
            .map_err(|e| GrpoError::Config(format!("the reward JSON schema is not valid: {e}")))?;
        Ok(Self {
            schema: Some(validator),
        })
    }

    /// [`with_schema`](Self::with_schema) from the schema's text.
    pub fn with_schema_text(text: &str) -> GrpoResult<Self> {
        let schema: Value = serde_json::from_str(text)
            .map_err(|e| GrpoError::Config(format!("the reward JSON schema is not JSON: {e}")))?;
        Self::with_schema(&schema)
    }

    fn passes(&self, completion: &str) -> bool {
        let Ok(document) = serde_json::from_str::<Value>(json_text(completion)) else {
            return false;
        };
        self.schema
            .as_ref()
            .is_none_or(|schema| schema.is_valid(&document))
    }
}

/// The text of the JSON document in `completion`: what is inside its first
/// fenced code block, if it has one, else the completion itself, trimmed.
fn json_text(completion: &str) -> &str {
    let trimmed = completion.trim();
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        // The opening fence's line may name the language.
        let body = after.split_once('\n').map_or("", |(_, body)| body);
        if let Some(end) = body.find("```") {
            return body[..end].trim();
        }
    }
    trimmed
}

impl RewardFunction for JsonReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        Ok(completions
            .iter()
            .map(|completion| f64::from(self.passes(completion)))
            .collect())
    }

    fn name(&self) -> &str {
        if self.schema.is_some() {
            "json_schema"
        } else {
            "json"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> String {
        text.to_string()
    }

    #[test]
    fn the_regex_reward_matches_anywhere_unless_anchored() {
        let any = RegexReward::new(r"\d+").unwrap();
        let got = any
            .compute(&[], &[s("the answer is 42"), s("no digits")], None)
            .unwrap();
        assert_eq!(got, vec![1.0, 0.0]);

        let whole = RegexReward::new(r"^\d+$").unwrap();
        let got = whole
            .compute(&[], &[s("42"), s("the answer is 42")], None)
            .unwrap();
        assert_eq!(got, vec![1.0, 0.0]);

        let err = RegexReward::new("(unclosed").unwrap_err();
        assert!(matches!(err, GrpoError::Config(m) if m.contains("(unclosed")));
    }

    #[test]
    fn the_json_reward_wants_a_document_and_nothing_around_it() {
        let reward = JsonReward::any();
        let got = reward
            .compute(
                &[],
                &[
                    s(r#"{"a": 1}"#),
                    s("  [1, 2]\n"),
                    s("```json\n{\"a\": 1}\n```"),
                    s("Here you go: {\"a\": 1}"),
                    s("{\"a\": "),
                    s(""),
                ],
                None,
            )
            .unwrap();
        assert_eq!(got, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_json_reward_validates_against_a_schema() {
        let reward = JsonReward::with_schema_text(
            r#"{"type": "object", "required": ["name", "age"],
                "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}}}"#,
        )
        .unwrap();
        let got = reward
            .compute(
                &[],
                &[
                    s(r#"{"name": "Ada", "age": 36}"#),
                    s(r#"{"name": "Ada"}"#),
                    s(r#"{"name": "Ada", "age": -1}"#),
                    s(r#"{"name": 7, "age": 3}"#),
                    s("```json\n{\"name\": \"Ada\", \"age\": 36}\n```"),
                    s("not json"),
                ],
                None,
            )
            .unwrap();
        assert_eq!(got, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(reward.name(), "json_schema");
    }

    #[test]
    fn a_bad_schema_is_a_config_error() {
        assert!(matches!(
            JsonReward::with_schema_text("not json"),
            Err(GrpoError::Config(_))
        ));
        assert!(matches!(
            JsonReward::with_schema_text(r#"{"type": 12}"#),
            Err(GrpoError::Config(_))
        ));
    }
}
