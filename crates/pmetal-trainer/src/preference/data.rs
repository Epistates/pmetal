//! Loading preference datasets.
//!
//! Rows come from JSONL, a JSON array, or Parquet (so a Hugging Face dataset
//! resolved by `resolve_dataset_path` loads directly). Two shapes are read:
//!
//! - **Paired**: `prompt`, `chosen`, `rejected`.
//! - **Unpaired** (KTO): `prompt`, `completion`, `label` (a boolean, a number
//!   above zero, or a word like `"good"`). Paired rows also load as KTO data:
//!   the chosen completion is a desirable example and the rejected one an
//!   undesirable example.
//!
//! Each of `prompt`, `chosen`, `rejected` and `completion` is either a string
//! or a list of `{"role", "content"}` messages. When a `chosen` / `rejected`
//! list carries the whole conversation and there is no `prompt`, every message
//! before its last assistant turn is the prompt. Common alternative field
//! names (`instruction`, `question`, `accepted`, `response`, ...) are accepted.
//!
//! With a chat template, the prompt is rendered as the conversation up to the
//! assistant's turn and the completion as that turn, end-of-turn token
//! included, exactly as supervised fine-tuning renders it. Without one, prompt
//! and completion are tokenized as plain text and the completion ends in EOS.

use std::io::BufRead;
use std::path::Path;

use pmetal_data::Tokenizer;
use pmetal_data::chat_templates::{ChatTemplate, Message, TrainingSampleBuilder};
use serde_json::Value;

use super::{KtoSample, PreferencePair, Sequence};

const PROMPT_FIELDS: &[&str] = &["prompt", "instruction", "question", "input", "query"];
const CHOSEN_FIELDS: &[&str] = &[
    "chosen",
    "accepted",
    "preferred",
    "chosen_response",
    "output_chosen",
];
const REJECTED_FIELDS: &[&str] = &[
    "rejected",
    "dispreferred",
    "rejected_response",
    "output_rejected",
];
const COMPLETION_FIELDS: &[&str] = &["completion", "response", "output", "answer"];
const LABEL_FIELDS: &[&str] = &["label", "desirable", "rating", "thumbs_up"];

/// Read every row of a JSONL, JSON-array or Parquet file as a JSON object.
pub fn read_rows(path: impl AsRef<Path>) -> anyhow::Result<Vec<Value>> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if ext == "parquet" {
        return read_parquet_rows(path);
    }
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("can't open {}: {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut first = String::new();
    while first.trim().is_empty() {
        if reader.read_line(&mut first)? == 0 {
            return Ok(Vec::new());
        }
    }
    if first.trim_start().starts_with('[') {
        let mut rest = String::new();
        std::io::Read::read_to_string(&mut reader, &mut rest)?;
        first.push_str(&rest);
        return Ok(serde_json::from_str(&first)?);
    }
    let mut rows = vec![serde_json::from_str(first.trim())?];
    for (n, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(
            serde_json::from_str(&line)
                .map_err(|e| anyhow::anyhow!("{} line {}: {e}", path.display(), n + 2))?,
        );
    }
    Ok(rows)
}

fn read_parquet_rows(path: &Path) -> anyhow::Result<Vec<Value>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut writer = arrow::json::ArrayWriter::new(Vec::new());
    for batch in reader {
        writer.write(&batch?)?;
    }
    writer.finish()?;
    let bytes = writer.into_inner();
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

/// A field's value: plain text, or a list of chat messages.
enum Turns {
    Text(String),
    Messages(Vec<Message>),
}

fn field<'a>(row: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names
        .iter()
        .find_map(|n| row.get(*n).filter(|v| !v.is_null()))
}

fn turns(value: &Value) -> Option<Turns> {
    match value {
        Value::String(s) => Some(Turns::Text(s.clone())),
        Value::Array(items) => items
            .iter()
            .map(|m| {
                let role = m.get("role").or_else(|| m.get("from"))?.as_str()?;
                let content = m.get("content").or_else(|| m.get("value"))?.as_str()?;
                let role = match role {
                    "human" => "user",
                    "gpt" | "bot" => "assistant",
                    other => other,
                };
                Some(Message::new(role, content))
            })
            .collect::<Option<Vec<_>>>()
            .map(Turns::Messages),
        _ => None,
    }
}

/// Split a completion into the conversation before it and its text.
fn completion(value: &Value) -> Option<(Vec<Message>, String)> {
    match turns(value)? {
        Turns::Text(text) => Some((Vec::new(), text)),
        Turns::Messages(mut messages) => {
            let last = messages.pop()?;
            (last.role == "assistant").then_some((messages, last.content))
        }
    }
}

/// The prompt messages for a row whose completion is `completion_prefix` +
/// assistant turn: the `prompt` field if there is one, otherwise the
/// completion's own preceding turns. A `system` field is prepended.
fn prompt_messages(row: &Value, completion_prefix: Vec<Message>) -> Option<Vec<Message>> {
    let mut messages = match field(row, PROMPT_FIELDS).map(turns) {
        Some(Some(Turns::Text(text))) => {
            let mut m = vec![Message::user(text)];
            m.extend(completion_prefix);
            m
        }
        Some(Some(Turns::Messages(mut m))) => {
            m.extend(completion_prefix);
            m
        }
        Some(None) => return None,
        None if !completion_prefix.is_empty() => completion_prefix,
        None => return None,
    };
    if let Some(system) = row.get("system").and_then(Value::as_str) {
        if !system.is_empty() && messages.first().is_none_or(|m| m.role != "system") {
            messages.insert(0, Message::system(system));
        }
    }
    Some(messages)
}

/// How sequences are tokenized and how long they may be.
struct Encoder<'a> {
    tokenizer: &'a Tokenizer,
    template: Option<&'a ChatTemplate>,
    max_prompt_len: usize,
    max_len: usize,
}

impl Encoder<'_> {
    /// Tokenize `prompt` + an assistant reply of `reply`, then fit it: the
    /// prompt keeps its last `max_prompt_len` tokens and the whole sequence
    /// its first `max_len`. `None` when no completion token survives.
    fn encode(&self, prompt: &[Message], reply: &str) -> anyhow::Result<Option<Sequence>> {
        let (ids, labels) = match self.template {
            Some(template) => {
                let mut messages = prompt.to_vec();
                messages.push(Message::assistant(reply));
                TrainingSampleBuilder::new(template.clone()).build_tokenized(
                    &messages,
                    self.tokenizer,
                    usize::MAX,
                )?
            }
            None => {
                let text: String = prompt
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let prompt_ids = self.tokenizer.encode_with_special_tokens(&text)?;
                let mut reply_ids = self.tokenizer.encode(reply)?;
                if let Some(eos) = self.tokenizer.eos_token_id() {
                    reply_ids.push(eos);
                }
                let s = Sequence::new(&prompt_ids, &reply_ids);
                (s.ids, s.labels)
            }
        };
        let prompt_len = labels.iter().position(|&l| l != -100).unwrap_or(ids.len());
        let skip = prompt_len.saturating_sub(self.max_prompt_len);
        let end = ids.len().min(skip + self.max_len);
        let seq = Sequence {
            ids: ids[skip..end].to_vec(),
            labels: labels[skip..end].to_vec(),
        };
        Ok((seq.completion_len() > 0).then_some(seq))
    }
}

/// Load (prompt, chosen, rejected) pairs.
///
/// Rows that are missing a field, or whose completion doesn't survive
/// truncation, are skipped with a warning; it is an error if none are left.
pub fn load_preference_pairs(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    template: Option<&ChatTemplate>,
    max_prompt_len: usize,
    max_len: usize,
) -> anyhow::Result<Vec<PreferencePair>> {
    let path = path.as_ref();
    let rows = read_rows(path)?;
    let enc = Encoder {
        tokenizer,
        template,
        max_prompt_len,
        max_len,
    };
    let mut pairs = Vec::with_capacity(rows.len());
    let mut skipped = 0usize;
    for row in &rows {
        match paired_row(&enc, row)? {
            Some(pair) => pairs.push(pair),
            None => skipped += 1,
        }
    }
    if skipped > 0 {
        tracing::warn!(
            "{}: skipped {skipped} of {} rows without a usable prompt, chosen and rejected",
            path.display(),
            rows.len()
        );
    }
    if pairs.is_empty() {
        anyhow::bail!(
            "{}: no preference pairs (rows need prompt/chosen/rejected)",
            path.display()
        );
    }
    Ok(pairs)
}

fn paired_row(enc: &Encoder<'_>, row: &Value) -> anyhow::Result<Option<PreferencePair>> {
    let (Some((chosen_prefix, chosen)), Some((rejected_prefix, rejected))) = (
        field(row, CHOSEN_FIELDS).and_then(completion),
        field(row, REJECTED_FIELDS).and_then(completion),
    ) else {
        return Ok(None);
    };
    // Whole-conversation rows repeat the prompt in `rejected`; when the two
    // disagree the row isn't a pair.
    let same = |a: &[Message], b: &[Message]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.role == y.role && x.content == y.content)
    };
    if !rejected_prefix.is_empty() && !same(&rejected_prefix, &chosen_prefix) {
        return Ok(None);
    }
    let Some(prompt) = prompt_messages(row, chosen_prefix) else {
        return Ok(None);
    };
    match (
        enc.encode(&prompt, &chosen)?,
        enc.encode(&prompt, &rejected)?,
    ) {
        (Some(chosen), Some(rejected)) => Ok(Some(PreferencePair { chosen, rejected })),
        _ => Ok(None),
    }
}

fn parse_label(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|v| v > 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" | "good" | "desirable" | "chosen" | "preferred" | "positive"
            | "thumbs_up" | "up" => Some(true),
            "false" | "no" | "0" | "-1" | "bad" | "undesirable" | "rejected" | "negative"
            | "thumbs_down" | "down" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Load labelled completions for KTO.
///
/// Each example's KL partner is the next example's completion under its own
/// prompt (the last wraps to the first), which is how KTO estimates the
/// reference point from mismatched pairs.
pub fn load_kto_samples(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    template: Option<&ChatTemplate>,
    max_prompt_len: usize,
    max_len: usize,
) -> anyhow::Result<Vec<KtoSample>> {
    let path = path.as_ref();
    let rows = read_rows(path)?;
    let enc = Encoder {
        tokenizer,
        template,
        max_prompt_len,
        max_len,
    };
    // (prompt, completion text, desirable)
    let mut examples: Vec<(Vec<Message>, String, bool)> = Vec::with_capacity(rows.len());
    let mut skipped = 0usize;
    for row in &rows {
        if let Some((prefix, text)) = field(row, COMPLETION_FIELDS).and_then(completion) {
            let label = field(row, LABEL_FIELDS).and_then(parse_label);
            match (prompt_messages(row, prefix), label) {
                (Some(prompt), Some(label)) => examples.push((prompt, text, label)),
                _ => skipped += 1,
            }
            continue;
        }
        let chosen = field(row, CHOSEN_FIELDS).and_then(completion);
        let rejected = field(row, REJECTED_FIELDS).and_then(completion);
        match (chosen, rejected) {
            (Some((prefix, chosen)), Some((_, rejected))) => match prompt_messages(row, prefix) {
                Some(prompt) => {
                    examples.push((prompt.clone(), chosen, true));
                    examples.push((prompt, rejected, false));
                }
                None => skipped += 1,
            },
            _ => skipped += 1,
        }
    }

    let mut samples = Vec::with_capacity(examples.len());
    for (i, (prompt, text, desirable)) in examples.iter().enumerate() {
        let partner = &examples[(i + 1) % examples.len()].1;
        match (enc.encode(prompt, text)?, enc.encode(prompt, partner)?) {
            (Some(sequence), Some(mismatched)) => samples.push(KtoSample {
                sequence,
                mismatched,
                desirable: *desirable,
            }),
            _ => skipped += 1,
        }
    }
    if skipped > 0 {
        tracing::warn!(
            "{}: skipped {skipped} rows without a usable prompt, completion and label",
            path.display()
        );
    }
    if samples.is_empty() {
        anyhow::bail!(
            "{}: no KTO examples (rows need prompt/completion/label, or prompt/chosen/rejected)",
            path.display()
        );
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn conversational_rows_split_prompt_from_the_last_assistant_turn() {
        let row = json!({
            "chosen": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ],
            "rejected": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "go away"}
            ]
        });
        let (prefix, chosen) = field(&row, CHOSEN_FIELDS).and_then(completion).unwrap();
        assert_eq!(chosen, "hello");
        let prompt = prompt_messages(&row, prefix).unwrap();
        assert_eq!(prompt.len(), 1);
        assert_eq!(
            (prompt[0].role.as_str(), prompt[0].content.as_str()),
            ("user", "hi")
        );
    }

    #[test]
    fn string_rows_take_the_prompt_field_and_system() {
        let row = json!({"system": "be brief", "question": "2+2?", "chosen": "4", "rejected": "5"});
        let (prefix, chosen) = field(&row, CHOSEN_FIELDS).and_then(completion).unwrap();
        assert_eq!(chosen, "4");
        let prompt = prompt_messages(&row, prefix).unwrap();
        let roles: Vec<&str> = prompt.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user"]);
    }

    #[test]
    fn a_completion_must_end_on_an_assistant_turn() {
        assert!(completion(&json!([{"role": "user", "content": "x"}])).is_none());
        assert!(completion(&json!(42)).is_none());
    }

    #[test]
    fn labels_parse_from_bools_numbers_and_words() {
        assert_eq!(parse_label(&json!(true)), Some(true));
        assert_eq!(parse_label(&json!(0)), Some(false));
        assert_eq!(parse_label(&json!(3)), Some(true));
        assert_eq!(parse_label(&json!("Thumbs_Down")), Some(false));
        assert_eq!(parse_label(&json!("maybe")), None);
    }

    #[test]
    fn rows_read_from_jsonl_and_json_arrays() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("d.jsonl");
        std::fs::write(&jsonl, "{\"a\":1}\n\n{\"a\":2}\n").unwrap();
        assert_eq!(read_rows(&jsonl).unwrap().len(), 2);
        let arr = dir.path().join("d.json");
        std::fs::write(&arr, "[{\"a\":1},{\"a\":2},{\"a\":3}]").unwrap();
        assert_eq!(read_rows(&arr).unwrap().len(), 3);
    }
}
