//! Prompt datasets for GRPO.
//!
//! GRPO samples its own completions, so a row contributes a *prompt*: the
//! conversation up to where the model starts writing, rendered with the
//! model's own chat template and its generation prompt (the assistant header
//! the model continues from), exactly as `pmetal infer` and `pmetal serve`
//! render one. A row's answer is not part of the prompt: it is kept apart as
//! the [`reference`](GrpoPrompt::reference) that verifiable rewards compare
//! against.
//!
//! Rows are read from JSONL, JSON, Parquet, Arrow or CSV (see
//! [`read_rows`](crate::preference::read_rows)). A prompt is a `prompt`
//! (also `question`, `instruction`, `input`, `query`) given as a string or a
//! list of `{"role", "content"}` messages, with an optional `system`; or a
//! `messages` list, whose last assistant turn, if any, is the reference
//! rather than part of the prompt. The reference is the first of `answer`,
//! `solution`, `reference`, `ground_truth`, `target`, `completion`,
//! `response`, `output` the row has; a GSM8K-style `... #### 18` keeps what
//! follows the last `####`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pmetal_data::Tokenizer;
use pmetal_data::chat_templates::{ChatTemplate, Message};
use serde_json::Value;

use super::DatasetRow;
use crate::preference::data::{Turns, field, prompt_messages, read_rows, turns};

const REFERENCE_FIELDS: &[&str] = &[
    "answer",
    "solution",
    "reference",
    "ground_truth",
    "target",
    "completion",
    "response",
    "output",
];

/// One prompt to roll out from.
#[derive(Debug, Clone, PartialEq)]
pub struct GrpoPrompt {
    /// The prompt's tokens, generation prompt included.
    pub input_ids: Vec<u32>,
    /// The row's reference answer, if it has one.
    pub reference: Option<String>,
    /// Images the prompt refers to (vision-language models).
    pub images: Option<Vec<PathBuf>>,
    /// The row's other columns, as the dataset had them: what an external or
    /// scripted reward scores a completion against (see
    /// [`RewardBatch::columns`](super::RewardBatch::columns)). The prompt's
    /// own columns and the images are not among them.
    pub columns: DatasetRow,
}

impl GrpoPrompt {
    /// A prompt of token ids, with no reference, images or columns.
    pub fn new(input_ids: Vec<u32>) -> Self {
        Self {
            input_ids,
            reference: None,
            images: None,
            columns: DatasetRow::default(),
        }
    }
}

/// The columns of `row` a reward may read: all of them but the ones that make
/// up the prompt (the conversation, however the dataset names it) and the
/// images, which a reward gets as tensors if it gets them at all.
fn reward_columns(row: &Value, columns: &PromptColumns) -> DatasetRow {
    const PROMPT_KEYS: &[&str] = &["prompt", "messages", "conversations", "images", "image"];
    let Value::Object(map) = row else {
        return DatasetRow::default();
    };
    let prompt_column = columns.prompt.as_deref();
    Arc::new(
        map.iter()
            .filter(|(key, _)| {
                !PROMPT_KEYS.contains(&key.as_str()) && Some(key.as_str()) != prompt_column
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

/// The reference a row names: text as it is (after a GSM8K-style `####`),
/// numbers as written, a message list as its last assistant turn.
fn reference_of(row: &Value) -> Option<String> {
    let value = field(row, REFERENCE_FIELDS)?;
    let text = match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => match turns(other)? {
            Turns::Messages(messages) => messages.last()?.content.clone(),
            Turns::Text(text) => text,
        },
    };
    let text = match text.rsplit_once("####") {
        Some((_, answer)) => answer.trim().to_string(),
        None => text.trim().to_string(),
    };
    (!text.is_empty()).then_some(text)
}

/// The prompt messages of a row and the reference it carries, or `None` for a
/// row with no prompt.
fn prompt_and_reference(row: &Value) -> Option<(Vec<Message>, Option<String>)> {
    if let Some(Turns::Messages(mut messages)) =
        field(row, &["messages", "conversations"]).and_then(turns)
    {
        let mut reference = reference_of(row);
        if messages.last().is_some_and(|m| m.role == "assistant") {
            let answer = messages.pop().map(|m| m.content);
            reference = reference.or(answer);
        }
        if let Some(system) = row.get("system").and_then(Value::as_str) {
            if !system.is_empty() && messages.first().is_none_or(|m| m.role != "system") {
                messages.insert(0, Message::system(system));
            }
        }
        return (!messages.is_empty()).then_some((messages, reference));
    }
    let messages = prompt_messages(row, Vec::new())?;
    Some((messages, reference_of(row)))
}

fn image_paths(row: &Value, base: &Path) -> Option<Vec<PathBuf>> {
    let value = field(row, &["images", "image"])?;
    let names: Vec<&str> = match value {
        Value::String(s) => vec![s.as_str()],
        Value::Array(items) => items.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    let paths: Vec<PathBuf> = names
        .into_iter()
        .map(|name| {
            let path = Path::new(name);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                base.join(path)
            }
        })
        .collect();
    (!paths.is_empty()).then_some(paths)
}

/// Tokenize one row's prompt: the chat template's inference rendering
/// (generation prompt included), as `pmetal infer` tokenizes it.
fn encode_prompt(
    messages: &[Message],
    template: &ChatTemplate,
    tokenizer: &Tokenizer,
) -> anyhow::Result<Vec<u32>> {
    let text = template.apply_inference(messages, false, None).text;
    Ok(tokenizer.encode_with_special_tokens(&text)?)
}

/// Which columns hold the prompt and the reference answer, when they are not
/// among the names the loader looks for.
#[derive(Debug, Clone, Default)]
pub struct PromptColumns {
    /// The prompt's column.
    pub prompt: Option<String>,
    /// The reference answer's column.
    pub reference: Option<String>,
}

impl PromptColumns {
    /// `row` with the named columns moved to the names the loader reads.
    fn normalize(&self, row: &Value) -> Value {
        let (Value::Object(map), true) = (row, self.prompt.is_some() || self.reference.is_some())
        else {
            return row.clone();
        };
        let mut map = map.clone();
        if let Some(value) = self.prompt.as_ref().and_then(|c| map.get(c)).cloned() {
            map.remove("messages");
            map.remove("conversations");
            map.insert("prompt".into(), value);
        }
        if let Some(value) = self.reference.as_ref().and_then(|c| map.get(c)).cloned() {
            map.insert("answer".into(), value);
        }
        Value::Object(map)
    }
}

/// Load the prompts of the dataset at `path` (see the module docs for the row
/// formats). Rows without a usable prompt are skipped with a warning; it is
/// an error if none are left.
pub fn load_prompts(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    max_prompt_len: usize,
) -> anyhow::Result<Vec<GrpoPrompt>> {
    load_prompts_with(
        path,
        tokenizer,
        template,
        max_prompt_len,
        &PromptColumns::default(),
    )
}

/// [`load_prompts`] reading the prompt and reference from the columns `columns` names.
pub fn load_prompts_with(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    max_prompt_len: usize,
    columns: &PromptColumns,
) -> anyhow::Result<Vec<GrpoPrompt>> {
    let path = path.as_ref();
    let base = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let rows = read_rows(path)?;
    let mut prompts = Vec::with_capacity(rows.len());
    let mut skipped = 0usize;
    let mut truncated = 0usize;
    for raw in &rows {
        let row = columns.normalize(raw);
        let row = &row;
        let Some((messages, reference)) = prompt_and_reference(row) else {
            skipped += 1;
            continue;
        };
        let ids = encode_prompt(&messages, template, tokenizer)?;
        let ids = if ids.len() > max_prompt_len {
            // The end of the prompt is the generation prompt: keep it.
            truncated += 1;
            ids[ids.len() - max_prompt_len..].to_vec()
        } else {
            ids
        };
        if ids.is_empty() {
            skipped += 1;
            continue;
        }
        prompts.push(GrpoPrompt {
            input_ids: ids,
            reference,
            images: image_paths(row, &base),
            columns: reward_columns(raw, columns),
        });
    }
    if skipped > 0 {
        tracing::warn!(
            "{}: skipped {skipped} of {} rows without a prompt",
            path.display(),
            rows.len()
        );
    }
    if truncated > 0 {
        tracing::warn!(
            "{}: {truncated} prompts are longer than {max_prompt_len} tokens and keep their last \
             {max_prompt_len}",
            path.display()
        );
    }
    if prompts.is_empty() {
        anyhow::bail!(
            "{}: no prompts (rows need a `prompt`, `question` or `messages` column)",
            path.display()
        );
    }
    let with_reference = prompts.iter().filter(|p| p.reference.is_some()).count();
    tracing::info!(
        "{}: {} prompts, {with_reference} with a reference answer",
        path.display(),
        prompts.len()
    );
    Ok(prompts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_prompt_row_keeps_its_answer_out_of_the_prompt() {
        let row = json!({"question": "what is 6*7?", "answer": "so it is\n#### 42"});
        let (messages, reference) = prompt_and_reference(&row).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "what is 6*7?");
        assert_eq!(reference.as_deref(), Some("42"));
    }

    #[test]
    fn a_conversation_ending_on_the_assistant_gives_the_reference() {
        let row = json!({
            "system": "be brief",
            "messages": [
                {"role": "user", "content": "2+2?"},
                {"role": "assistant", "content": "4"}
            ]
        });
        let (messages, reference) = prompt_and_reference(&row).unwrap();
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user"]);
        assert_eq!(reference.as_deref(), Some("4"));

        // A conversation that ends on the user has no reference.
        let row = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(prompt_and_reference(&row).unwrap().1, None);
    }

    #[test]
    fn numeric_references_and_rows_without_prompts() {
        let row = json!({"prompt": "x?", "answer": 18});
        assert_eq!(prompt_and_reference(&row).unwrap().1.as_deref(), Some("18"));
        assert!(prompt_and_reference(&json!({"answer": "4"})).is_none());
        assert!(prompt_and_reference(&json!({"messages": []})).is_none());
    }

    #[test]
    fn named_columns_stand_in_for_the_usual_ones() {
        let columns = PromptColumns {
            prompt: Some("problem".into()),
            reference: Some("final".into()),
        };
        // The row also has the usual names, which the named columns override.
        let row = json!({
            "problem": "what is 6*7?", "final": "42",
            "answer": "wrong", "messages": [{"role": "user", "content": "other"}]
        });
        let (messages, reference) = prompt_and_reference(&columns.normalize(&row)).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "what is 6*7?");
        assert_eq!(reference.as_deref(), Some("42"));
    }

    #[test]
    fn a_reward_sees_every_column_but_the_prompt_and_the_images() {
        let row = json!({
            "prompt": "p", "messages": [], "images": ["a.png"], "problem": "q?",
            "answer": "#### 42", "tests": ["assert f(1) == 2"], "id": 7
        });
        let columns = PromptColumns {
            prompt: Some("problem".into()),
            reference: None,
        };
        let got = reward_columns(&row, &columns);
        let mut keys: Vec<&str> = got.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["answer", "id", "tests"]);
        // The columns are the row's as it was, not the loader's reading of it.
        assert_eq!(got["answer"], "#### 42");
        assert!(reward_columns(&json!("not a row"), &columns).is_empty());
    }

    #[test]
    fn image_paths_resolve_against_the_dataset_directory() {
        let row = json!({"prompt": "p", "images": ["a.png", "/abs/b.png"]});
        let got = image_paths(&row, Path::new("/data")).unwrap();
        assert_eq!(
            got,
            vec![PathBuf::from("/data/a.png"), PathBuf::from("/abs/b.png")]
        );
        assert!(image_paths(&json!({"prompt": "p"}), Path::new("/data")).is_none());
    }

    /// A word-level tokenizer over the ChatML markers and a few words.
    fn tokenizer() -> Tokenizer {
        let words = [
            "[UNK]",
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
            "what",
            "is",
            "6",
            "*",
            "7",
            "?",
            "42",
        ];
        let vocab: serde_json::Map<String, Value> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), json!(i)))
            .collect();
        let added: Vec<Value> = ["<|im_start|>", "<|im_end|>"]
            .iter()
            .map(|w| {
                json!({"id": words.iter().position(|x| x == w).unwrap(), "content": w,
                       "special": true, "single_word": false, "lstrip": false,
                       "rstrip": false, "normalized": false})
            })
            .collect();
        let tokenizer = json!({
            "version": "1.0",
            "added_tokens": added,
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        });
        Tokenizer::from_bytes(tokenizer.to_string().as_bytes()).expect("tokenizer parses")
    }

    /// The prompt is the conversation up to the assistant's turn, generation
    /// prompt included, and the answer is nowhere in it.
    #[test]
    fn prompts_end_at_the_generation_prompt_and_exclude_the_answer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.jsonl");
        std::fs::write(
            &path,
            "{\"prompt\": \"what is 6 * 7 ?\", \"answer\": \"42\"}\n{\"nothing\": 1}\n",
        )
        .unwrap();
        let tokenizer = tokenizer();
        let template = ChatTemplate::chatml();
        let prompts = load_prompts(&path, &tokenizer, &template, 64).unwrap();
        assert_eq!(prompts.len(), 1);
        let ids = &prompts[0].input_ids;
        let id = |w: &str| tokenizer.encode(w).unwrap()[0];
        // "<|im_start|> user ... <|im_end|> <|im_start|> assistant", and not "42".
        assert_eq!(ids[0], id("<|im_start|>"));
        assert_eq!(*ids.last().unwrap(), id("assistant"));
        assert!(!ids.contains(&id("42")));
        assert_eq!(prompts[0].reference.as_deref(), Some("42"));

        // A prompt longer than the limit keeps its end, the generation prompt.
        let cut = load_prompts(&path, &tokenizer, &template, 3).unwrap();
        assert_eq!(cut[0].input_ids.len(), 3);
        assert_eq!(*cut[0].input_ids.last().unwrap(), id("assistant"));
    }
}
