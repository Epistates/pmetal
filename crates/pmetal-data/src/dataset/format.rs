//! Dataset formats: what a row's columns say it is, and how to read it.
//!
//! Every container (JSONL, Parquet, Arrow, ...) yields [`Row`]s, and a row
//! reads as one of three things, whatever its format:
//!
//! - **text**: a document, trained whole;
//! - **completion**: a prompt and what answers it, the answer trained;
//! - **chat**: a conversation, the assistant's turns trained.

use crate::chat_templates::{ContentPart, Message, ToolCall, ToolDefinition};
use crate::rows::Row;
use pmetal_core::{PMetalError, Result};
use serde_json::Value;

/// Dataset format variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetFormat {
    /// Simple format: `{"text": "..."}` (or `{"content": "..."}`), trained whole.
    Simple,
    /// Alpaca format: `{"instruction": "...", "input": "...", "output": "..."}`;
    /// the output is trained.
    Alpaca,
    /// ShareGPT format: `{"conversations": [{"from": "human", "value": "..."}, ...]}`.
    ShareGpt,
    /// OpenAI format: `{"messages": [{"role": "user", "content": "..."}, ...]}`,
    /// with tool calls, tool results, content parts and per-message `weight`.
    OpenAi,
    /// Reasoning format: `{"problem": "...", "thinking": "...", "solution": "..."}`.
    Reasoning,
    /// Prompt and completion: `{"prompt": "...", "completion": "..."}`, each a
    /// string or a list of messages; only the completion is trained.
    PromptCompletion,
    /// Detect the format from the first row's columns (see
    /// [`DatasetFormat::detect`]).
    Auto,
    /// Custom column extraction — use arbitrary JSON field names.
    Custom {
        /// Field name for the text content (or combined prompt+response).
        text_column: String,
        /// Multiple columns to concatenate (e.g. `["thinking", "solution"]`).
        /// Takes precedence over `text_column` when non-empty.
        text_columns: Option<Vec<String>>,
        /// Separator between concatenated columns (default: `"\n\n"`).
        column_separator: String,
        /// Optional: field for the prompt portion (masked from loss).
        prompt_column: Option<String>,
        /// Optional: separate response column.
        response_column: Option<String>,
    },
}

/// Configuration for custom column extraction from a dataset's rows.
///
/// Use this to specify which columns contain the training text, and optionally
/// which column contains the prompt (for loss masking) vs the response.
#[derive(Debug, Clone, Default)]
pub struct DatasetColumnConfig {
    /// Main text/content column name. When set and differs from "text", overrides
    /// auto-detection and uses custom column extraction.
    pub text_column: Option<String>,
    /// Multiple text columns to concatenate (e.g. `["thinking", "solution"]`).
    /// Joined with `column_separator` between each. Takes precedence over `text_column`.
    pub text_columns: Option<Vec<String>>,
    /// Separator between concatenated columns (default: `"\n\n"`).
    pub column_separator: Option<String>,
    /// Prompt column (masked from loss). When set, only response tokens contribute
    /// to the training loss.
    pub prompt_column: Option<String>,
    /// Response column. When set together with `prompt_column`, the full sequence
    /// is `prompt || response` and the prompt portion receives label -100.
    pub response_column: Option<String>,
}

/// Columns that can hold the prompt of a prompt/completion row.
const PROMPT_FIELDS: &[&str] = &["prompt", "question", "query", "input"];
/// Columns that can hold the completion of a prompt/completion row.
const COMPLETION_FIELDS: &[&str] = &["completion", "response", "output", "answer"];

/// What a conversation's rows carry besides its messages.
#[derive(Debug, Clone, Default)]
pub(crate) struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    /// Per message: `Some(false)` keeps an assistant turn out of the loss
    /// (a prompt's few-shot turn, a `weight` of 0); `None` leaves it to the
    /// role.
    pub trained: Vec<Option<bool>>,
}

/// A row, read.
#[derive(Debug, Clone)]
pub(crate) enum Example {
    /// A document, trained whole.
    Text(String),
    /// A prompt and its completion; `text = prompt + completion`.
    Completion { prompt: String, completion: String },
    /// A conversation.
    Chat(Conversation),
}

fn invalid(message: impl Into<String>) -> PMetalError {
    PMetalError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

/// The value under `key`, unless it is absent or null.
fn field<'a>(row: &'a Row, key: &str) -> Option<&'a Value> {
    row.get(key).filter(|v| !v.is_null())
}

/// The first of `keys` the row has a value for.
fn first_field<'a>(row: &'a Row, keys: &[&'a str]) -> Option<(&'a str, &'a Value)> {
    keys.iter().find_map(|k| field(row, k).map(|v| (*k, v)))
}

fn column_names(row: &Row) -> String {
    let names: Vec<&str> = row.keys().map(String::as_str).collect();
    format!("[{}]", names.join(", "))
}

/// A scalar JSON value as text: strings as they are, numbers and booleans
/// printed. Anything structured is not text.
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// A required text column.
fn required_text(row: &Row, key: &str, what: &str) -> Result<String> {
    field(row, key).and_then(scalar_text).ok_or_else(|| {
        invalid(format!(
            "{what} needs a text column '{key}'; the row has {}",
            column_names(row)
        ))
    })
}

impl DatasetFormat {
    /// The format a row's columns say it is.
    ///
    /// The structured layouts are tried first, so a column that merely
    /// restates them (the pre-rendered `text` that `tatsu-lab/alpaca` ships
    /// beside `instruction`, `input` and `output`) does not hide the split
    /// between prompt and answer: `messages`, then `conversations`, then
    /// `instruction` with an answer, then `problem` / `thinking`, then a
    /// prompt column (`prompt`, `question`, `query`, `input`) with a
    /// completion column (`completion`, `response`, `output`, `answer`), and
    /// `text` or `content` last.
    ///
    /// # Errors
    /// If none match; the message lists the row's columns. Pass the format
    /// explicitly, or a [`DatasetColumnConfig`], to read other layouts.
    pub fn detect(row: &Row) -> Result<Self> {
        let has = |key: &str| field(row, key).is_some();
        let is_array = |key: &str| field(row, key).is_some_and(Value::is_array);
        let completion = first_field(row, COMPLETION_FIELDS).is_some();

        Ok(if is_array("messages") {
            Self::OpenAi
        } else if is_array("conversations") {
            Self::ShareGpt
        } else if has("instruction") && completion {
            Self::Alpaca
        } else if has("problem") || has("thinking") {
            Self::Reasoning
        } else if first_field(row, PROMPT_FIELDS).is_some() && completion {
            Self::PromptCompletion
        } else if has("text") || has("content") {
            Self::Simple
        } else {
            return Err(invalid(format!(
                "Could not detect the dataset format from its columns {}. Expected 'messages', \
                 'conversations', 'instruction' + 'output', 'prompt' + 'completion', \
                 'problem', or 'text'; pass the format or the column names explicitly.",
                column_names(row)
            )));
        })
    }

    /// Read one row in this format.
    ///
    /// `Auto` is resolved against this row; callers that read many rows
    /// resolve it once, on the first.
    pub(crate) fn parse(&self, row: &Row) -> Result<Example> {
        match self {
            Self::Auto => Self::detect(row)?.parse(row),
            Self::Simple => {
                let (key, _) = first_field(row, &["text", "content"]).ok_or_else(|| {
                    invalid(format!(
                        "Expected a 'text' column; the row has {}",
                        column_names(row)
                    ))
                })?;
                Ok(Example::Text(required_text(row, key, "A text row")?))
            }
            Self::Alpaca => parse_alpaca(row),
            Self::OpenAi => parse_chat(row, "messages"),
            Self::ShareGpt => parse_chat(row, "conversations"),
            Self::Reasoning => parse_reasoning(row),
            Self::PromptCompletion => parse_prompt_completion(row),
            Self::Custom {
                text_column,
                text_columns,
                column_separator,
                prompt_column,
                response_column,
            } => parse_custom(
                row,
                text_column,
                text_columns.as_deref(),
                column_separator,
                prompt_column.as_deref(),
                response_column.as_deref(),
            ),
        }
    }
}

/// The format `columns` ask for, if they ask for one.
///
/// Columns named after the reasoning layout (`thinking` / `solution` with a
/// `problem` prompt) read as [`DatasetFormat::Reasoning`], which writes the
/// `<think>` tags and masks the problem; any other column choice is
/// [`DatasetFormat::Custom`]. A text column of the default name `text`, alone,
/// asks for nothing.
pub(crate) fn effective_format(
    format: DatasetFormat,
    columns: Option<&DatasetColumnConfig>,
) -> DatasetFormat {
    let Some(cols) = columns else {
        return format;
    };
    let multi = cols.text_columns.as_ref().is_some_and(|v| !v.is_empty());
    let custom_text = cols.text_column.as_ref().is_some_and(|t| t != "text");

    let reasoning_pattern = multi
        && cols.prompt_column.as_deref() == Some("problem")
        && cols
            .text_columns
            .as_ref()
            .is_some_and(|c| c.iter().any(|c| c == "thinking" || c == "solution"));
    if reasoning_pattern {
        tracing::info!(
            "Detected reasoning-pattern columns (thinking/solution + problem); \
             using Reasoning format for proper <think> tags and loss masking"
        );
        return DatasetFormat::Reasoning;
    }
    if multi || custom_text || cols.prompt_column.is_some() || cols.response_column.is_some() {
        return DatasetFormat::Custom {
            text_column: cols
                .text_column
                .clone()
                .unwrap_or_else(|| "text".to_string()),
            text_columns: cols.text_columns.clone(),
            column_separator: cols
                .column_separator
                .clone()
                .unwrap_or_else(|| "\n\n".to_string()),
            prompt_column: cols.prompt_column.clone(),
            response_column: cols.response_column.clone(),
        };
    }
    format
}

// -- text and completions ------------------------------------------------------

fn parse_alpaca(row: &Row) -> Result<Example> {
    let instruction = required_text(row, "instruction", "An Alpaca row")?;
    let input = field(row, "input")
        .and_then(scalar_text)
        .unwrap_or_default();
    let (_, output) = first_field(row, &["output", "response"]).ok_or_else(|| {
        invalid(format!(
            "An Alpaca row needs an 'output'; the row has {}",
            column_names(row)
        ))
    })?;
    let completion = scalar_text(output)
        .ok_or_else(|| invalid("An Alpaca row's output is not text".to_string()))?;
    let prompt = if input.is_empty() {
        format!("### Instruction:\n{instruction}\n\n### Response:\n")
    } else {
        format!("### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
    };
    Ok(Example::Completion { prompt, completion })
}

fn parse_reasoning(row: &Row) -> Result<Example> {
    let text = |key: &str| field(row, key).and_then(scalar_text).unwrap_or_default();
    let (problem, thinking, solution) = (text("problem"), text("thinking"), text("solution"));
    let completion = if thinking.is_empty() {
        solution
    } else {
        format!("<think>\n{thinking}\n</think>\n\n{solution}")
    };
    Ok(Example::Completion {
        prompt: format!("{problem}\n\n"),
        completion,
    })
}

fn parse_prompt_completion(row: &Row) -> Result<Example> {
    let (_, prompt) = first_field(row, PROMPT_FIELDS).ok_or_else(|| {
        invalid(format!(
            "A prompt row needs a 'prompt'; the row has {}",
            column_names(row)
        ))
    })?;
    let (_, completion) = first_field(row, COMPLETION_FIELDS).ok_or_else(|| {
        invalid(format!(
            "A prompt row needs a 'completion'; the row has {}",
            column_names(row)
        ))
    })?;

    if let (Some(prompt), Some(completion)) = (scalar_text(prompt), scalar_text(completion)) {
        return Ok(Example::Completion { prompt, completion });
    }
    // Conversational: either side is a list of messages.
    let mut conversation = Conversation::default();
    let mut add = |value: &Value, role: &str, trained: Option<bool>| -> Result<()> {
        let messages = match value {
            Value::Array(items) => parse_messages(items)?,
            other => vec![ParsedMessage::plain(
                role,
                scalar_text(other).ok_or_else(|| {
                    invalid("a prompt or completion is neither text nor messages")
                })?,
            )],
        };
        for m in messages {
            conversation.trained.push(trained.or(m.trained));
            conversation.messages.push(m.message);
        }
        Ok(())
    };
    add(prompt, "user", Some(false))?;
    add(completion, "assistant", None)?;
    conversation.tools = parse_tools(row)?;
    Ok(Example::Chat(conversation))
}

/// Custom column extraction; see [`DatasetColumnConfig`].
fn parse_custom(
    row: &Row,
    text_col: &str,
    text_cols: Option<&[String]>,
    separator: &str,
    prompt_col: Option<&str>,
    response_col: Option<&str>,
) -> Result<Example> {
    let prompt_of = |col: &str| field(row, col).and_then(Value::as_str).map(str::to_string);

    let (text, prompt) = if let Some(cols) = text_cols.filter(|c| !c.is_empty()) {
        let parts: Vec<String> = cols
            .iter()
            .filter_map(|c| field(row, c).map(extract_text))
            .collect();
        if parts.is_empty() {
            return Err(invalid(format!(
                "None of the requested columns {cols:?} were present in this row; \
                 the row has {}",
                column_names(row)
            )));
        }
        let mut text = parts.join(separator);
        let prompt = prompt_col.and_then(prompt_of);
        // A prompt column that is not one of the text columns is not in the
        // text yet: it leads it.
        if let (Some(pc), Some(prompt_text)) = (prompt_col, prompt.as_deref()) {
            if !cols.iter().any(|c| c == pc) {
                text = format!("{prompt_text}{separator}{text}");
            }
        }
        (text, prompt)
    } else if let (Some(pc), Some(rc)) = (prompt_col, response_col) {
        let prompt = prompt_of(pc).unwrap_or_default();
        let response = field(row, rc).map(extract_text).unwrap_or_default();
        (format!("{prompt}{response}"), Some(prompt))
    } else {
        let text = field(row, text_col).map(extract_text).ok_or_else(|| {
            invalid(format!(
                "Column '{text_col}' not found; the row has {}",
                column_names(row)
            ))
        })?;
        (text, prompt_col.and_then(prompt_of))
    };

    match prompt {
        Some(prompt) if !prompt.is_empty() && text.starts_with(&prompt) => {
            // The separator after the prompt is part of the prompt: it is
            // what the model is shown before it answers.
            let mut split = prompt.len();
            if !separator.is_empty() && text[split..].starts_with(separator) {
                split += separator.len();
            }
            Ok(Example::Completion {
                prompt: text[..split].to_string(),
                completion: text[split..].to_string(),
            })
        }
        _ => Ok(Example::Text(text)),
    }
}

/// Text of a column that holds text, or a chat message list, or something
/// else (serialized).
fn extract_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            // Chat format: [{"role": "user", "content": "..."}, ...]
            let parts: Vec<String> = items
                .iter()
                .filter_map(|item| {
                    let content = item
                        .get("content")
                        .or_else(|| item.get("value"))?
                        .as_str()?;
                    Some(
                        match item
                            .get("role")
                            .or_else(|| item.get("from"))
                            .and_then(Value::as_str)
                        {
                            Some(role) => format!("{role}: {content}"),
                            None => content.to_string(),
                        },
                    )
                })
                .collect();
            if parts.is_empty() {
                value.to_string()
            } else {
                parts.join("\n")
            }
        }
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

// -- chat -----------------------------------------------------------------------

/// A parsed message and whether its row keeps it out of the loss.
struct ParsedMessage {
    message: Message,
    trained: Option<bool>,
}

impl ParsedMessage {
    fn plain(role: &str, content: String) -> Self {
        Self {
            message: Message::new(canonical_role(role), content),
            trained: None,
        }
    }
}

/// The role a dataset's name for it means.
fn canonical_role(role: &str) -> String {
    match role.trim().to_ascii_lowercase().as_str() {
        "human" | "user" | "customer" => "user".to_string(),
        "gpt" | "assistant" | "chatgpt" | "bot" | "model" | "agent" | "function_call" => {
            "assistant".to_string()
        }
        "system" => "system".to_string(),
        "tool" | "function" | "observation" | "ipython" => "tool".to_string(),
        other => other.to_string(),
    }
}

/// A message's content: text, a list of parts, nothing (a tool call carries
/// none), or JSON (a tool result).
fn parse_content(value: Option<&Value>) -> (String, Option<Vec<ContentPart>>) {
    match value {
        None | Some(Value::Null) => (String::new(), None),
        Some(Value::String(s)) => (s.clone(), None),
        Some(Value::Array(items)) => {
            let parts: Vec<ContentPart> = items.iter().filter_map(content_part).collect();
            let media = parts.iter().any(|p| !matches!(p, ContentPart::Text(_)));
            if media {
                let message = Message::with_parts("user", parts);
                (message.content, message.parts)
            } else {
                let text = parts
                    .into_iter()
                    .filter_map(|p| match p {
                        ContentPart::Text(t) => Some(t),
                        _ => None,
                    })
                    .collect::<String>();
                (text, None)
            }
        }
        Some(other) => (other.to_string(), None),
    }
}

/// One item of a content list; `None` for an item that is not text or media.
fn content_part(item: &Value) -> Option<ContentPart> {
    if let Some(text) = item.as_str() {
        return Some(ContentPart::Text(text.to_string()));
    }
    let kind = item.get("type").and_then(Value::as_str);
    match kind {
        Some("image" | "image_url" | "input_image") => Some(ContentPart::Image),
        Some("video" | "video_url" | "input_video") => Some(ContentPart::Video),
        _ => item
            .get("text")
            .and_then(Value::as_str)
            .map(|t| ContentPart::Text(t.to_string())),
    }
}

/// Tool calls as the OpenAI API writes them, or flat (`{"name", "arguments"}`),
/// in a list or a JSON string.
fn parse_tool_calls(value: &Value) -> Result<Vec<ToolCall>> {
    let items = match value {
        Value::Array(items) => items.clone(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(items)) => items,
            _ => return Err(invalid("tool_calls is a string that is not a JSON list")),
        },
        _ => return Err(invalid("tool_calls is not a list")),
    };
    items
        .into_iter()
        .map(|mut item| {
            if item.get("function").is_none() && item.get("name").is_some() {
                item = serde_json::json!({
                    "id": item.get("id").cloned().unwrap_or(Value::Null),
                    "function": {
                        "name": item["name"],
                        "arguments": item.get("arguments").cloned().unwrap_or(Value::Null),
                    },
                });
            }
            serde_json::from_value::<ToolCall>(item)
                .map_err(|e| invalid(format!("a tool call is not valid: {e}")))
        })
        .collect()
}

/// The `tools` column: OpenAI tool definitions or bare function definitions,
/// in a list or a JSON string.
fn parse_tools(row: &Row) -> Result<Vec<ToolDefinition>> {
    let items = match field(row, "tools") {
        None => return Ok(Vec::new()),
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(s)) if s.trim().is_empty() => return Ok(Vec::new()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(items)) => items,
            _ => return Err(invalid("tools is a string that is not a JSON list")),
        },
        Some(_) => return Err(invalid("tools is not a list")),
    };
    items
        .into_iter()
        .map(|item| {
            let item = if item.get("function").is_none() && item.get("name").is_some() {
                serde_json::json!({"type": "function", "function": item})
            } else {
                item
            };
            serde_json::from_value::<ToolDefinition>(item)
                .map_err(|e| invalid(format!("a tool definition is not valid: {e}")))
        })
        .collect()
}

/// Read a list of messages, in OpenAI keys (`role` / `content`) or ShareGPT's
/// (`from` / `value`).
fn parse_messages(items: &[Value]) -> Result<Vec<ParsedMessage>> {
    items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let obj = item
                .as_object()
                .ok_or_else(|| invalid(format!("message {} is not an object", i + 1)))?;
            let role = obj
                .get("role")
                .or_else(|| obj.get("from"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("message {} has no role", i + 1)))?;
            let (content, parts) = parse_content(obj.get("content").or_else(|| obj.get("value")));
            let mut message = Message::new(canonical_role(role), content);
            message.parts = parts;
            if let Some(calls) = obj.get("tool_calls").filter(|v| !v.is_null()) {
                let calls = parse_tool_calls(calls)
                    .map_err(|e| invalid(format!("message {}: {e}", i + 1)))?;
                if !calls.is_empty() {
                    message.tool_calls = Some(calls);
                }
            }
            message.tool_call_id = obj
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            message.reasoning_content = obj
                .get("reasoning_content")
                .or_else(|| obj.get("reasoning"))
                .and_then(Value::as_str)
                .map(str::to_string);
            // OpenAI's chat fine-tuning format: `weight: 0` keeps an
            // assistant turn out of the loss.
            let trained = (message.role == "assistant"
                && obj.get("weight").and_then(Value::as_f64) == Some(0.0))
            .then_some(false);
            Ok(ParsedMessage { message, trained })
        })
        .collect()
}

/// A conversation row: its messages under `key`, a `system` column leading
/// them when the messages hold no system turn, and its `tools`.
fn parse_chat(row: &Row, key: &str) -> Result<Example> {
    let items = field(row, key).and_then(Value::as_array).ok_or_else(|| {
        invalid(format!(
            "Expected a list under '{key}'; the row has {}",
            column_names(row)
        ))
    })?;
    let mut conversation = Conversation::default();
    if let Some(system) = field(row, "system").and_then(Value::as_str) {
        let has_system = items.iter().any(|m| {
            matches!(
                m.get("role")
                    .or_else(|| m.get("from"))
                    .and_then(Value::as_str),
                Some("system")
            )
        });
        if !has_system && !system.is_empty() {
            conversation.messages.push(Message::system(system));
            conversation.trained.push(None);
        }
    }
    for m in parse_messages(items)? {
        conversation.trained.push(m.trained);
        conversation.messages.push(m.message);
    }
    conversation.tools = parse_tools(row)?;
    Ok(Example::Chat(conversation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(value: Value) -> Row {
        value.as_object().unwrap().clone()
    }

    fn detect(value: Value) -> DatasetFormat {
        DatasetFormat::detect(&row(value)).unwrap()
    }

    #[test]
    fn structured_columns_win_over_a_text_column() {
        assert_eq!(
            detect(json!({"text": "x", "instruction": "i", "output": "o"})),
            DatasetFormat::Alpaca
        );
        assert_eq!(
            detect(json!({"text": "x", "messages": [{"role": "user", "content": "q"}]})),
            DatasetFormat::OpenAi
        );
        assert_eq!(
            detect(json!({"text": "x", "prompt": "p", "completion": "c"})),
            DatasetFormat::PromptCompletion
        );
    }

    #[test]
    fn each_layout_is_detected() {
        assert_eq!(detect(json!({"text": "x"})), DatasetFormat::Simple);
        assert_eq!(detect(json!({"content": "x"})), DatasetFormat::Simple);
        assert_eq!(
            detect(json!({"conversations": [{"from": "human", "value": "q"}]})),
            DatasetFormat::ShareGpt
        );
        assert_eq!(
            detect(json!({"problem": "p", "solution": "s"})),
            DatasetFormat::Reasoning
        );
        assert_eq!(
            detect(json!({"question": "q", "answer": "a"})),
            DatasetFormat::PromptCompletion
        );
        assert_eq!(
            detect(json!({"input": "q", "output": "a"})),
            DatasetFormat::PromptCompletion
        );
        // A null column is no column.
        assert_eq!(
            detect(json!({"instruction": null, "text": "x"})),
            DatasetFormat::Simple
        );
    }

    #[test]
    fn a_row_nothing_matches_names_its_columns() {
        let err =
            DatasetFormat::detect(&row(json!({"chosen": "a", "rejected": "b", "prompt": "p"})))
                .unwrap_err()
                .to_string();
        assert!(err.contains("chosen, rejected, prompt"), "{err}");
    }

    #[test]
    fn alpaca_input_may_be_null_missing_or_empty() {
        for input in [json!(null), json!(""), Value::Null] {
            let r = row(json!({"instruction": "i", "input": input, "output": "o"}));
            let Example::Completion { prompt, completion } =
                DatasetFormat::Alpaca.parse(&r).unwrap()
            else {
                panic!("a completion")
            };
            assert_eq!(prompt, "### Instruction:\ni\n\n### Response:\n");
            assert_eq!(completion, "o");
        }
        let r = row(json!({"instruction": "i", "output": 4}));
        assert!(DatasetFormat::Alpaca.parse(&r).is_ok(), "a number is text");
    }

    #[test]
    fn messages_read_parts_tool_calls_tools_and_weights() {
        let r = row(json!({
            "system": "sys",
            "tools": "[{\"name\": \"f\", \"description\": \"d\", \"parameters\": {}}]",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "look "}, {"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "assistant", "content": null, "weight": 0, "tool_calls": [
                    {"name": "f", "arguments": {"a": 1}}]},
                {"role": "tool", "tool_call_id": "c", "content": {"ok": true}},
                {"role": "assistant", "content": "done", "reasoning_content": "hmm"}]
        }));
        let Example::Chat(c) = DatasetFormat::OpenAi.parse(&r).unwrap() else {
            panic!("a chat")
        };
        let roles: Vec<&str> = c.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool", "assistant"]);
        assert!(c.messages[1].has_media());
        assert_eq!(c.messages[1].content, "look ");
        assert_eq!(
            c.messages[2].tool_calls.as_ref().unwrap()[0].function.name,
            "f"
        );
        assert_eq!(c.messages[3].content, "{\"ok\":true}");
        assert_eq!(c.messages[4].reasoning_content.as_deref(), Some("hmm"));
        assert_eq!(c.tools.len(), 1);
        assert_eq!(c.trained, [None, None, Some(false), None, None]);
    }

    #[test]
    fn sharegpt_roles_map_to_chat_roles() {
        let r = row(json!({"conversations": [
            {"from": "system", "value": "s"}, {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"}, {"from": "function_call", "value": "{}"},
            {"from": "observation", "value": "r"}]}));
        let Example::Chat(c) = DatasetFormat::ShareGpt.parse(&r).unwrap() else {
            panic!("a chat")
        };
        let roles: Vec<&str> = c.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "assistant", "tool"]);
    }

    #[test]
    fn a_conversational_prompt_keeps_its_turns_out_of_the_loss() {
        let r = row(json!({
            "prompt": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "shot"}],
            "completion": [{"role": "assistant", "content": "a"}]}));
        let Example::Chat(c) = DatasetFormat::PromptCompletion.parse(&r).unwrap() else {
            panic!("a chat")
        };
        assert_eq!(c.trained, [Some(false), Some(false), None]);
        // A text prompt with a message completion.
        let r = row(json!({"prompt": "q", "completion": [{"role": "assistant", "content": "a"}]}));
        let Example::Chat(c) = DatasetFormat::PromptCompletion.parse(&r).unwrap() else {
            panic!("a chat")
        };
        assert_eq!(c.messages[0].role, "user");
    }

    #[test]
    fn custom_columns_keep_the_separator_with_the_prompt() {
        let r = row(json!({"q": "Why?", "think": "because", "a": "so"}));
        let format = DatasetFormat::Custom {
            text_column: "text".into(),
            text_columns: Some(vec!["think".into(), "a".into()]),
            column_separator: "\n\n".into(),
            prompt_column: Some("q".into()),
            response_column: None,
        };
        let Example::Completion { prompt, completion } = format.parse(&r).unwrap() else {
            panic!("a completion")
        };
        assert_eq!(prompt, "Why?\n\n");
        assert_eq!(completion, "because\n\nso");

        let r = row(json!({"p": "2+2=", "r": " 4"}));
        let format = DatasetFormat::Custom {
            text_column: "text".into(),
            text_columns: None,
            column_separator: "\n\n".into(),
            prompt_column: Some("p".into()),
            response_column: Some("r".into()),
        };
        let Example::Completion { prompt, completion } = format.parse(&r).unwrap() else {
            panic!("a completion")
        };
        assert_eq!((prompt.as_str(), completion.as_str()), ("2+2=", " 4"));
    }

    #[test]
    fn column_flags_ask_for_a_custom_format_only_when_they_name_something() {
        let none = DatasetColumnConfig::default();
        assert_eq!(
            effective_format(DatasetFormat::Auto, Some(&none)),
            DatasetFormat::Auto
        );
        let text = DatasetColumnConfig {
            text_column: Some("text".into()),
            ..Default::default()
        };
        assert_eq!(
            effective_format(DatasetFormat::Auto, Some(&text)),
            DatasetFormat::Auto
        );
        let with_prompt = DatasetColumnConfig {
            text_column: Some("text".into()),
            prompt_column: Some("prompt".into()),
            ..Default::default()
        };
        assert!(matches!(
            effective_format(DatasetFormat::Auto, Some(&with_prompt)),
            DatasetFormat::Custom { .. }
        ));
        let reasoning = DatasetColumnConfig {
            text_columns: Some(vec!["thinking".into(), "solution".into()]),
            prompt_column: Some("problem".into()),
            ..Default::default()
        };
        assert_eq!(
            effective_format(DatasetFormat::Auto, Some(&reasoning)),
            DatasetFormat::Reasoning
        );
    }
}
